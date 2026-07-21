#!/usr/bin/env python3
"""
build_corpus.py
===============
Builds a balanced multilingual corpus of cooking recipes and non-recipe
book passages.

Target per language:
    1000 recipe passages
    1000 non-recipe (book) passages
    => exact 50/50 recipe / non-recipe split.

One "sample" = one recipe OR one book passage.

Languages with no native dataset fall back to their closest sibling
(Catalan->Spanish, Venetian->Italian, Old Danish->Latin/German family, etc.).

IMPORTANT - read before running
--------------------------------
The requested languages live in completely different ecosystems and there is
NO single source that serves all of them. This script handles the sources that
expose a real programmatic endpoint:

  * RecipeNLG  ......... English recipes (and English-sibling fallback).
                         Requires a local CSV (see RECIPENLG_CSV below); it is
                         a Kaggle download and cannot be auto-fetched without
                         credentials. Streamed/capped, never loaded whole.
  * Hugging Face ....... modern DE / FR / ES / IT recipe sets, STREAMED so the
                         full dataset is never downloaded.
  * CoReMA (GAMS) ...... medieval DE / LA / FR recipes, fetched as per-object
                         plaintext, capped at the per-language target.
  * Gutendex/Gutenberg . non-recipe books per language, split into passages.

Anything with no real endpoint (Venetian, Catalan, Old Danish, Middle Dutch,
Middle Low German, etc.) is routed to a sibling. If even the sibling has no
endpoint, the slot is logged as a MANUAL TODO rather than crashing the run.

Nothing here downloads a whole dataset: every fetcher streams or paginates and
stops once the per-language target is met.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PER_LANG_TARGET = 1000          # recipes per language == non-recipes per language
MIN_PASSAGE_WORDS = 40          # a "book passage" must be at least this long
MAX_PASSAGE_WORDS = 250         # ...and is truncated to at most this long
OUTPUT_DIR = Path("corpus_out")
CACHE_DIR = Path("corpus_cache")
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.3    # be polite to public endpoints
USER_AGENT = "corpus-builder/1.0 (research; contact: you@example.org)"

random.seed(13)
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ----------------------------------------------------------------------------
# Source descriptors
# ----------------------------------------------------------------------------

@dataclass
class Source:
    """Describes how to obtain samples of one kind for one logical language."""
    kind: str                         # "recipe" or "book"
    fetcher: str                      # name of fetcher function
    params: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class LangPlan:
    """How to fill the recipe and book slots for a single language."""
    name: str
    recipe: Optional[Source]
    book: Optional[Source]
    sibling_of: Optional[str] = None  # for logging only


# ----------------------------------------------------------------------------
# Language -> source resolution
# ----------------------------------------------------------------------------
# Gutendex uses ISO-639-1 codes for the `languages` filter. Many historical
# variants share their modern parent's Gutenberg pool, so we map them there.

GUTENDEX_LANG = {
    "english": "en", "early modern english": "en",
    "german": "de", "old german": "de",
    "middle low german": "de", "middle dutch": "nl",
    "french": "fr", "old french": "fr", "middle french": "fr",
    "italian": "it", "venetian/italian": "it", "venetian": "it",
    "spanish": "es", "catalan": "ca",
    "latin": "la", "old danish": "da",
    "multiple": "en",  # mixed-language bucket: default to English books
}

# Modern recipe datasets on Hugging Face that can be STREAMED.
# (dataset_id, config, split, text_columns)  -- verify availability at runtime.
HF_RECIPE_SETS = {
    # de/fr/it served by Wikibooks (see WIKIBOOKS_RECIPE_BOOKS below)
    "es": ("Frorozcol/recetas-cocina", None, "train", ["steps", "ingredients"]),
    # mbien/recipe_nlg and m3hrdadfi/recipe_nlg_lite use legacy dataset scripts
    # removed in datasets>=3; Shengtao/recipe is a parquet-native replacement.
    "en": ("Shengtao/recipe", None, "train", ["directions", "description"]),
}

# Wikibooks cooking books for modern languages lacking a working HF dataset.
# (wikibooks_lang, page_prefix)
WIKIBOOKS_RECIPE_BOOKS = {
    "de": ("de", "Kochbuch"),            # ~860 single-level recipe pages
    "it": ("it", "Libro di cucina/Ricette"),   # ~767 recipe pages
    "fr": ("fr", "Livre de cuisine"),    # ~1300 single-level recipe pages
}

# Sent Soví (Llibre de Sent Soví) — oldest preserved Catalan cookbook (~72 recipes).
# The blog at sentsovi.cat publishes each recipe with its original medieval Catalan text
# under a "Text original del Sent Soví" heading. We scrape that section only.
SENTSOVI_INDEX = "http://www.sentsovi.cat/rubriques-index-de-receptes/"
SENTSOVI_BASE  = "http://www.sentsovi.cat"

# CoReMA medieval recipe base URL.
# Discovery via the project's Solr API is broken; we enumerate known manuscript
# IDs directly. All accessible manuscripts are in German-family languages.
COREMA_BASE = "https://gams.uni-graz.at"
COREMA_MANUSCRIPTS = [
    "b1", "b2", "b3", "b4", "b5", "b6",
    "br1", "bs1", "bs2", "db1", "ds1",
    "er1", "er2", "gr1", "k1", "m11",
    "pa1", "so1", "w1", "w4",
    "wo1a", "wo1b", "wo1c", "wo3b",
    "wo7a", "wo7b", "wo8",
    "wo9a", "wo9b",
    "wo10a", "wo10b", "wo10c",
    "wo11a", "wo11b", "wo11c",
]

# Sibling fallbacks: language -> language to borrow from when no native source.
SIBLING = {
    "catalan": "spanish",
    "venetian/italian": "italian",
    "venetian": "italian",
    "old danish": "latin",          # closest available medieval recipe family
    "middle dutch": "german",
    "middle low german": "german",
    "old german": "german",
    "old french": "french",
    "middle french": "french",
    "early modern english": "english",
    "multiple": "english",
}

# The full requested language list (deduplicated, order preserved).
REQUESTED_LANGUAGES = [
    "english", "old german", "catalan", "german", "multiple", "italian",
    "latin", "french", "old french", "venetian/italian", "old danish",
    "middle dutch", "middle low german", "early modern english",
    "middle french", "spanish",
]


def resolve_recipe_source(lang: str) -> tuple[Optional[Source], Optional[str]]:
    """Return (Source, sibling_used) for recipes in `lang`."""
    code = GUTENDEX_LANG.get(lang)
    # 1. Latin -> De Re Coquinaria (Apicius, Project Gutenberg #16439)
    if lang == "latin":
        return Source("recipe", "fetch_apicius", {}), None
    # 1b. Catalan -> Sent Soví (sentsovi.cat, original medieval Catalan text)
    if lang == "catalan":
        return Source("recipe", "fetch_sent_sovi", {}), None
    # 2. Medieval families -> CoReMA (German-language manuscripts)
    if "old " in lang or "middle " in lang or lang == "old german":
        return Source("recipe", "fetch_corema", {}), None
    # 3. Modern languages with a Wikibooks cooking book
    if code in WIKIBOOKS_RECIPE_BOOKS:
        wb_lang, wb_prefix = WIKIBOOKS_RECIPE_BOOKS[code]
        return Source("recipe", "fetch_wikibooks",
                      {"lang": wb_lang, "book_prefix": wb_prefix}), None
    # 4. Modern languages with an HF recipe set (en, es)
    if code in HF_RECIPE_SETS:
        return Source("recipe", "fetch_hf_recipes", {"code": code}), None
    # 5. Sibling fallback
    sib = SIBLING.get(lang)
    if sib:
        src, _ = resolve_recipe_source(sib)
        return src, sib
    return None, None


def resolve_book_source(lang: str) -> tuple[Optional[Source], Optional[str]]:
    """Return (Source, sibling_used) for non-recipe books in `lang`."""
    code = GUTENDEX_LANG.get(lang)
    if code:
        return Source("book", "fetch_gutenberg", {"code": code}), None
    sib = SIBLING.get(lang)
    if sib:
        src, _ = resolve_book_source(sib)
        return src, sib
    return None, None


def build_plans() -> list[LangPlan]:
    plans = []
    for lang in REQUESTED_LANGUAGES:
        rsrc, rsib = resolve_recipe_source(lang)
        bsrc, bsib = resolve_book_source(lang)
        plans.append(LangPlan(lang, rsrc, bsrc, sibling_of=rsib or bsib))
    return plans


# ----------------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------------

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def to_passages(text: str) -> Iterator[str]:
    """Split a long text into book-passage-sized chunks by word count."""
    words = clean(text).split()
    for i in range(0, len(words), MAX_PASSAGE_WORDS):
        chunk = words[i:i + MAX_PASSAGE_WORDS]
        if len(chunk) >= MIN_PASSAGE_WORDS:
            yield " ".join(chunk)


# ----------------------------------------------------------------------------
# Fetchers  (each yields cleaned strings, lazily, and the caller stops at N)
# ----------------------------------------------------------------------------

def fetch_hf_recipes(params: dict) -> Iterator[str]:
    """Stream a Hugging Face recipe dataset (streaming=True -> no full DL)."""
    code = params["code"]
    spec = HF_RECIPE_SETS.get(code)
    if not spec:
        return
    ds_id, config, split, cols = spec
    try:
        from datasets import load_dataset
        ds = load_dataset(ds_id, config, split=split, streaming=True)
    except Exception as e:  # dataset id may not exist / be gated
        print(f"  [hf:{code}] '{ds_id}' unavailable ({e.__class__.__name__}) -> skip")
        return
    for row in ds:
        text = ""
        for c in cols:
            v = row.get(c)
            if v:
                text = v if isinstance(v, str) else " ".join(map(str, v))
                break
        text = clean(text)
        if len(text.split()) >= 10:
            yield text


def fetch_corema(params: dict, limit: int = PER_LANG_TARGET) -> Iterator[str]:
    """
    Fetch CoReMA medieval German recipes via the GAMS TEI viewer.

    The project's Solr discovery endpoint is unavailable, so we enumerate
    known manuscript IDs from COREMA_MANUSCRIPTS and probe recipe numbers
    1..200, stopping a manuscript when 10 consecutive 404s are seen.
    Text is extracted from the rendered HTML article element.
    """
    manuscripts = list(COREMA_MANUSCRIPTS)
    random.shuffle(manuscripts)
    for ms in manuscripts:
        consec_404 = 0
        for n in range(1, 201):
            url = f"{COREMA_BASE}/o:corema.{ms}.{n}/sdef:TEI/get"
            try:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                r = session.get(url, timeout=REQUEST_TIMEOUT)
                if r.status_code == 404:
                    consec_404 += 1
                    if consec_404 >= 10:
                        break
                    continue
                consec_404 = 0
                if not r.ok:
                    continue
                article_pos = r.text.find("<article")
                if article_pos == -1:
                    continue
                article_html = r.text[article_pos:article_pos + 20000]
                body = re.sub(r"<[^>]+>", " ", article_html)
                body = clean(body)
                if len(body.split()) >= 10:
                    yield body
            except Exception:
                continue


def fetch_apicius(params: dict, limit: int = PER_LANG_TARGET) -> Iterator[str]:
    """
    Fetch De Re Coquinaria (Apicius) from Project Gutenberg eBook #16439.
    The text is in Latin. Individual recipe entries are delimited by
    parenthesised numerals like (1), (2) … which we split on.
    """
    url = "https://www.gutenberg.org/cache/epub/16439/pg16439.txt"
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  [apicius] download failed ({e.__class__.__name__})")
        return

    text = strip_gutenberg_boilerplate(r.text)
    # Split on recipe-number markers "(N)" that begin a new recipe entry.
    chunks = re.split(r"\n\s*\(\d+\)\s*", text)
    for chunk in chunks:
        chunk = clean(chunk)
        if len(chunk.split()) >= 15:
            yield chunk


def fetch_wikibooks(params: dict, limit: int = PER_LANG_TARGET) -> Iterator[str]:
    """
    Fetch recipe pages from a Wikibooks cooking book.

    Page titles are listed via the MediaWiki allpages API, filtered to the
    direct children of the given prefix (one extra slash). Each page's
    wikitext is fetched, stripped of markup, and yielded as plain text.
    """
    lang = params["lang"]
    book_prefix = params["book_prefix"]
    api_url = f"https://{lang}.wikibooks.org/w/api.php"
    expected_depth = book_prefix.count("/") + 1

    # Collect all page titles under the prefix.
    titles: list[str] = []
    cont: dict = {}
    while True:
        try:
            r = session.get(api_url, params={
                "action": "query", "list": "allpages",
                "apprefix": book_prefix + "/", "aplimit": 500,
                "format": "json", **cont,
            }, timeout=30)
            if not r.ok:
                break
            data = r.json()
        except Exception as e:
            print(f"  [wikibooks:{lang}] listing failed ({e.__class__.__name__})")
            break
        titles.extend(
            p["title"] for p in data["query"]["allpages"]
            if p["title"].count("/") == expected_depth
        )
        if "continue" not in data:
            break
        cont = data["continue"]

    random.shuffle(titles)

    _wiki_clean = re.compile(
        r"\[\[(?:[^\|\]]*\|)?([^\]]*)\]\]"   # [[link|text]] → text
        r"|\{{2}[^\}]*\}{2}"                  # {{template}} → ''
        r"|=+\s*[^=\n]+\s*=+"                 # ==heading== → ''
        r"|'{2,}"                             # bold/italic markers
        r"|<[^>]+>"                           # HTML tags
    )

    for title in titles:
        try:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            r = session.get(api_url, params={
                "action": "parse", "page": title,
                "prop": "wikitext", "format": "json",
            }, timeout=30)
            if not r.ok:
                continue
            wikitext = r.json().get("parse", {}).get("wikitext", {}).get("*", "")
            text = _wiki_clean.sub(" ", wikitext)
            text = clean(text)
            if len(text.split()) >= 20:
                yield text
        except Exception:
            continue


def fetch_gutenberg(params: dict, limit: int = PER_LANG_TARGET) -> Iterator[str]:
    """
    Discover books in a language via Gutendex, download plaintext, and split
    into passages. Stops as soon as `limit` passages have been yielded by the
    caller, so typically only a handful of books are fetched.
    """
    code = params["code"]
    page_url = "https://gutendex.com/books/"
    page_params = {"languages": code, "mime_type": "text/plain"}
    fetched_passages = 0
    pages_seen = 0
    while page_url and fetched_passages < limit and pages_seen < 10:
        pages_seen += 1
        try:
            r = session.get(page_url, params=page_params if pages_seen == 1 else None,
                            timeout=90)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [gutenberg:{code}] listing failed ({e.__class__.__name__})")
            return
        for book in data.get("results", []):
            fmts = book.get("formats", {})
            txt_url = next(
                (u for m, u in fmts.items()
                 if m.startswith("text/plain") and not u.endswith(".zip")),
                None,
            )
            if not txt_url:
                continue
            try:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                br = session.get(txt_url, timeout=REQUEST_TIMEOUT)
                if not br.ok:
                    continue
                raw = strip_gutenberg_boilerplate(br.text)
                for passage in to_passages(raw):
                    yield passage
                    fetched_passages += 1
                    if fetched_passages >= limit:
                        return
            except Exception:
                continue
        page_url = data.get("next")


def fetch_sent_sovi(params: dict, limit: int = PER_LANG_TARGET) -> Iterator[str]:
    """
    Fetch original medieval Catalan recipe text from sentsovi.cat.

    Each post on the blog includes a section labelled "Text original del Sent Soví"
    that reproduces the original 14th-century Catalan text. We collect all recipe
    URLs from the recipe index and extract that section from each page.

    The Sent Soví contains ~72 recipes total, so this source will always be SHORT
    of the 1000-recipe target.  The site has a self-signed cert; we disable
    TLS verification for the HTTP requests.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Discover recipe URLs from the index page.
    recipe_urls: list[str] = []
    try:
        r = session.get(SENTSOVI_INDEX, timeout=REQUEST_TIMEOUT, verify=False)
        if r.ok:
            SKIP = {"xmlrpc", "documentacio", "about", "category", "tag",
                    "wp-", "feed", "sitemap", "rubriques", "#", "?"}
            seen: set[str] = set()
            for link in re.findall(r'href="(http://www\.sentsovi\.cat/[^"]+)"', r.text):
                path = link.replace(SENTSOVI_BASE + "/", "").strip("/")
                if path and "/" not in path and not any(x in path for x in SKIP):
                    if link not in seen:
                        seen.add(link)
                        recipe_urls.append(link)
    except Exception as e:
        print(f"  [sent_sovi] index fetch failed ({e.__class__.__name__})")

    if not recipe_urls:
        print("  [sent_sovi] no recipe URLs discovered")
        return

    random.shuffle(recipe_urls)
    ORIGINAL_MARKER = "Text original del Sent Soví"
    STOP_MARKERS = ["Deixa un comentari", "Cancel·la les respostes",
                    "L'adreça electrònica", "Comentaris"]

    for url in recipe_urls:
        try:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            r = session.get(url, timeout=REQUEST_TIMEOUT, verify=False)
            if not r.ok:
                continue
            # Strip all HTML tags
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"&[a-z#0-9]+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            # Extract only the original medieval Catalan section
            pos = text.find(ORIGINAL_MARKER)
            if pos == -1:
                continue
            excerpt = text[pos + len(ORIGINAL_MARKER):]
            # Cut at the first footer/comment marker
            for stop in STOP_MARKERS:
                idx = excerpt.find(stop)
                if idx != -1:
                    excerpt = excerpt[:idx]
            excerpt = clean(excerpt)
            if len(excerpt.split()) >= 15:
                yield excerpt
        except Exception:
            continue


def strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text


FETCHERS: dict[str, Callable[..., Iterator[str]]] = {
    "fetch_hf_recipes": fetch_hf_recipes,
    "fetch_corema": fetch_corema,
    "fetch_apicius": fetch_apicius,
    "fetch_wikibooks": fetch_wikibooks,
    "fetch_sent_sovi": fetch_sent_sovi,
    "fetch_gutenberg": fetch_gutenberg,
}


# ----------------------------------------------------------------------------
# Collection driver
# ----------------------------------------------------------------------------

def collect(source: Optional[Source], target: int, label: str) -> list[str]:
    """Run a source's fetcher and collect up to `target` unique samples."""
    if source is None:
        print(f"  [{label}] no source resolved -> MANUAL TODO")
        return []
    fetcher = FETCHERS[source.fetcher]
    seen: set[str] = set()
    out: list[str] = []
    _limit_fetchers = {"fetch_corema", "fetch_apicius", "fetch_wikibooks",
                       "fetch_sent_sovi", "fetch_gutenberg"}
    try:
        gen = (fetcher(source.params, limit=target)
               if source.fetcher in _limit_fetchers
               else fetcher(source.params))
    except TypeError:
        gen = fetcher(source.params)
    for sample in gen:
        h = hash(sample[:200])
        if h in seen:
            continue
        seen.add(h)
        out.append(sample)
        if len(out) >= target:
            break
    status = "OK" if len(out) >= target else "SHORT"
    print(f"  [{label}] collected {len(out)}/{target}  ({status})")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    plans = build_plans()
    manifest = []

    for plan in plans:
        sib = f"  (sibling: {plan.sibling_of})" if plan.sibling_of else ""
        print(f"\n=== {plan.name}{sib} ===")

        recipes = collect(plan.recipe, PER_LANG_TARGET, f"{plan.name}/recipe")
        books = collect(plan.book, PER_LANG_TARGET, f"{plan.name}/book")

        # Balance to 50/50 only when both sides have data; if one side is
        # missing entirely, keep the other side as-is rather than zeroing out.
        if recipes and books:
            n = min(len(recipes), len(books))
            if n < PER_LANG_TARGET:
                print(f"  [{plan.name}] balancing down to {n}/{n} "
                      f"(a side was short; raise sources to reach {PER_LANG_TARGET})")
            recipes, books = recipes[:n], books[:n]
        elif not recipes:
            print(f"  [{plan.name}] no recipes; writing {len(books)} book passages only")
        elif not books:
            print(f"  [{plan.name}] no books; writing {len(recipes)} recipe passages only")

        if not recipes and not books:
            print(f"  [{plan.name}] nothing collected, skipping file")
            manifest.append({
                "language": plan.name,
                "sibling_of": plan.sibling_of,
                "recipes": 0,
                "books": 0,
                "balanced_total": 0,
                "file": None,
            })
            continue

        safe = re.sub(r"[^a-z0-9]+", "_", plan.name.lower()).strip("_")
        out_path = OUTPUT_DIR / f"{safe}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recipes:
                f.write(json.dumps({"lang": plan.name, "label": "recipe",
                                    "text": r}, ensure_ascii=False) + "\n")
            for b in books:
                f.write(json.dumps({"lang": plan.name, "label": "book",
                                    "text": b}, ensure_ascii=False) + "\n")

        manifest.append({
            "language": plan.name,
            "sibling_of": plan.sibling_of,
            "recipes": len(recipes),
            "books": len(books),
            "balanced_total": len(recipes) + len(books),
            "file": str(out_path),
        })

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== SUMMARY =====")
    for m in manifest:
        print(f"{m['language']:<22} recipes={m['recipes']:>4} "
              f"books={m['books']:>4} total={m['balanced_total']:>4}")
    print(f"\nManifest -> {OUTPUT_DIR / 'manifest.json'}")
    print("Slots marked MANUAL TODO above need a source added by hand "
          "(no programmatic endpoint exists).")


if __name__ == "__main__":
    main()