import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

import pandas as pd
from tqdm import tqdm


# =========================================================
# Translation models
# =========================================================

MODEL_MAP = {
    "es": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
}


# =========================================================
# Translator
# =========================================================

class Translator:

    def __init__(self):
        self.models = {}

    def _load_model(self, lang):
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

    def translate(self, text, target_lang):
        import torch

        if target_lang not in self.models:
            self._load_model(target_lang)

        tokenizer, model, device = self.models[target_lang]

        batch = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(**batch)

        return tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )


    def translate_batch(self, texts, target_lang, batch_size=32):

        if target_lang not in self.models:
            self._load_model(target_lang)

        tokenizer, model, device = self.models[target_lang]

        outputs = []

        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i:i + batch_size]

            batch = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(**batch)

            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )

            outputs.extend(decoded)

        return outputs  

# =========================================================
# Balancing logic
# =========================================================
def balance_with_translation(
    df,
    translator,
    target_langs,
    text_col="text",
    label_col="label",
    lang_col="language",
):
    """
    Balance dataset by translating unique English rows
    into exactly ONE target language each.
    """

    if "en" not in df[lang_col].unique():
        raise ValueError("English rows are required.")

    augmented_rows = []

    # Current counts
    lang_counts = df[lang_col].value_counts().to_dict()

    max_count = max(lang_counts.values())

    english_df = (
        df[df[lang_col] == "en"]
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    used_indices = set()

    print("Current distribution:")
    print(lang_counts)

    for target_lang in tqdm(
        target_langs,
        desc="Languages",
        position=0,
    ):

        if target_lang == "en":
            continue

        current_count = lang_counts.get(target_lang, 0)

        needed = max_count - current_count

        if needed <= 0:
            print(f"{target_lang}: already balanced")
            continue

        print(f"{target_lang}: needs {needed} samples")

        available_df = english_df[
            ~english_df.index.isin(used_indices)
        ]

        if len(available_df) == 0:
            print("No unused English samples left.")
            break

        sampled_df = available_df.head(needed)

        used_indices.update(sampled_df.index.tolist())

        for _, row in tqdm(
            sampled_df.iterrows(),
            total=len(sampled_df),
            desc=f"Translating -> {target_lang}",
            leave=False,
            position=1,
        ):

            text = row[text_col]

            try:
                translated_text = translator.translate_batch(
                    [text],
                    target_lang
                )[0]

            except Exception as e:
                print(f"Translation error ({target_lang}): {e}")
                continue

            new_row = row.copy()

            new_row[text_col] = translated_text
            new_row[lang_col] = target_lang
            new_row["original_language"] = "en"
            new_row["is_translated"] = True

            augmented_rows.append(new_row)

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)

    return df

# =========================================================
# Config builder
# =========================================================

def construct_config(df):

    config = {
        "resources": {},
        "languages": {},
        "original_language": {},
        "size": len(df),
        "0": int((df["label"] == 0).sum()),
        "1": int((df["label"] == 1).sum()),
    }

    for _, row in df.iterrows():

        resource = row["resource"]
        language = row["language"]
        original_language = row["original_language"]
        label = int(row["label"])

        # resources
        config["resources"].setdefault(resource, 0)
        config["resources"][resource] += 1

        # languages
        if language not in config["languages"]:
            config["languages"][language] = {0: 0, 1: 0}

        config["languages"][language][label] += 1

        # original language
        if original_language not in config["original_language"]:
            config["original_language"][original_language] = {
                0: 0,
                1: 0,
            }

        config["original_language"][original_language][label] += 1

    return config


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--langs",
        nargs="+",
        required=True,
        help="Target languages e.g. es de fr",
    )

    args = parser.parse_args()

    input_path = Path(args.input_folder) / "corpus.parquet"

    df = pd.read_parquet(input_path)

    translator = Translator()

    balanced_df = balance_with_translation(
        df=df,
        translator=translator,
        target_langs=args.langs,
    )

    config = construct_config(balanced_df)

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    balanced_df.to_parquet(
        output_dir / "translated_corpus.parquet",
        index=False,
    )

    with open(
        output_dir / "translated_corpus_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("Done.")
    print(f"Final dataset size: {len(balanced_df)}")
    print(balanced_df["language"].value_counts())