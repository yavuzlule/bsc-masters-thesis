#!/usr/bin/env python3
"""
translate_llm.py

Batch translate all .txt files in a directory using cached models.
Models are loaded ONCE per language for performance.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Model registry
# ----------------------------
MODEL_MAP = {
    "es": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
}


# ----------------------------
# Translator with caching (CORE FIX)
# ----------------------------
class Translator:
    def __init__(self, backend: str = "transformers"):
        self.backend = backend
        self.models = {}  # lang -> (tokenizer, model)
        self.argos_cache = {}  # lang -> translator

    def _load_transformer(self, lang: str):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        if lang not in MODEL_MAP:
            raise ValueError(f"Unsupported language: {lang}")

        model_name = MODEL_MAP[lang]

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        self.models[lang] = (tokenizer, model, device)

    def translate(self, text: str, lang: str) -> str:
        if self.backend == "argos":
            return self._translate_argos(text, lang)
        return self._translate_transformers(text, lang)

    # ----------------------------
    # Transformers (cached)
    # ----------------------------
    def _translate_transformers(self, text: str, lang: str) -> str:
        import torch

        if lang not in self.models:
            self._load_transformer(lang)

        tokenizer, model, device = self.models[lang]

        batch = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            translated = model.generate(**batch)

        return tokenizer.decode(translated[0], skip_special_tokens=True)

    # ----------------------------
    # Argos (cached)
    # ----------------------------
    def _translate_argos(self, text: str, lang: str) -> str:
        import argostranslate.translate

        if lang not in self.argos_cache:
            installed_languages = argostranslate.translate.get_installed_languages()
            from_lang = next(l for l in installed_languages if l.code == "en")
            to_lang = next(l for l in installed_languages if l.code == lang)
            self.argos_cache[lang] = from_lang.get_translation(to_lang)

        return self.argos_cache[lang].translate(text)


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_chars: int = 800):
    paragraphs = text.split("\n")
    chunk = ""

    for p in paragraphs:
        if len(chunk) + len(p) < max_chars:
            chunk += p + "\n"
        else:
            yield chunk.strip()
            chunk = p + "\n"

    if chunk.strip():
        yield chunk.strip()


def translate_full(text: str, lang: str, translator: Translator) -> str:
    results = []
    for chunk in chunk_text(text):
        results.append(translator.translate(chunk, lang))
    return "\n".join(results)


# ----------------------------
# File I/O
# ----------------------------
def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ----------------------------
# Directory processing
# ----------------------------
def process_directory(input_dir: Path, out_dir: Path, langs, translator: Translator):
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print("No .txt files found.")
        return

    tasks = [(f, lang) for f in txt_files for lang in langs]

    for file_path, lang in tqdm(tasks, desc="Translating", unit="task"):
        text = read_file(file_path)

        translated = translate_full(text, lang, translator)

        output_name = f"{file_path.stem}_{lang}.txt"
        output_path = out_dir / output_name

        write_file(output_path, translated)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", default="translations")
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--backend", choices=["transformers", "argos"], default="transformers")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.outdir)

    translator = Translator(backend=args.backend)

    process_directory(input_dir, out_dir, args.langs, translator)


if __name__ == "__main__":
    main()