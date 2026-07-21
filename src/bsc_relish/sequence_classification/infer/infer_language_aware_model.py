from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file as load_safetensors

from bsc_relish.utils.models import LanguageAwareClassifier
from bsc_relish.utils.utils import get_device, load_config

language_order = [
    "english",
    "old german",
    "catalan",
    "multiple",
    "italian",
    "latin",
    "french",
    "old french",
    "venetian/italian",
    "old danish",
    "middle dutch",
    "middle low german",
    "early modern english",
    "middle french",
    "spanish",
    "german",
    "dutch",
]

UNK_ID = len(language_order)




def load_model(model_dir, base_model_name="bert-base-uncased"):
    """
    model_dir/
    ├── model_weights.pth
    ├── tokenizer files
    └── metrics.json
    """

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = LanguageAwareClassifier(
        model_name=base_model_name,
        num_labels=2,
        num_languages=len(language_order) + 1,
    )

    weights_path = Path(model_dir) / "model_weights.pth"

    state_dict = torch.load(
        weights_path,
        map_location=device
    )

    num_languages = state_dict[
        "lang_embedding.weight"
    ].shape[0]

    model = LanguageAwareClassifier(
        model_name=base_model_name,
        num_labels=2,
        num_languages=num_languages,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer, device

def prepare_language_ids(df):
    df = df.copy()

    df["language"] = (
        df["language"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    df["lang_id"] = df["language"].apply(
        lambda x: language_order.index(x)
        if x in language_order
        else UNK_ID
    )

    return df

def infer_language_aware(
    df,
    model_dir,
    base_model_name="bert-base-uncased",
    text_column="text",
    batch_size=32,
    max_length=256,
):
    model, tokenizer, device = load_model(
        model_dir,
        base_model_name,
    )

    df = prepare_language_ids(df)

    texts = (
        "[LANG="
        + df["language"].astype(str)
        + "] "
        + df[text_column].fillna("").astype(str)
    ).tolist()

    lang_ids = df["lang_id"].tolist()

    probabilities = []

    for start_idx in tqdm(
        range(0, len(texts), batch_size),
        desc="Inference",
    ):
        end_idx = min(start_idx + batch_size, len(texts))

        batch_texts = texts[start_idx:end_idx]
        batch_lang_ids = lang_ids[start_idx:end_idx]

        try:
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            batch_lang_ids = torch.tensor(
                batch_lang_ids,
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lang_ids=batch_lang_ids,
                )

            probs = torch.softmax(logits, dim=-1)

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )

        except Exception as e:
            print(
                f"Batch {start_idx}-{end_idx} failed: {e}"
            )
            probabilities.extend(
                [np.nan] * (end_idx - start_idx)
            )

    df["language_aware_proba"] = probabilities

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    model_name = config["model"]["name"]
    model_dir = (
        config["model"]["dir"]
    )

    data_path = (
        config["test-df"]
    )

    df = pd.read_parquet(data_path)

    df = infer_language_aware(
        df,
        model_dir=model_dir,
        base_model_name=model_name,
        batch_size=16,
        max_length=256,
    )

    run_id = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_path = (
        f"data/test/predictions-{model_name}-{run_id}.parquet"
    )

    df.to_parquet(output_path)

    print(f"Saved: {output_path}")

