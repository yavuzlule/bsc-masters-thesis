import pandas as pd

from bsc_relish.preprocess.ingest.label_loader import load_labeled_txt_files
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from bsc_relish.preprocess.embed.embed import add_embeddings
from bsc_relish.preprocess.io.store import DatasetStore
from sentence_transformers import SentenceTransformer

import argparse
import yaml




def run_pipeline(config):
    # 1. ingestion
    rows = load_labeled_txt_files(
        root_dir="data/raw",
        label_map=config["data"]["label_map"],
        max_files=config["data"]["limit_files"]["max_files"]
    )

    # 2. chunking
    if config["preprocessing"]["chunking"]["enabled"]:
        rows = expand_chunks(
            rows,
            max_words=config["preprocessing"]["chunking"]["max_words"]
        )
        text_column = "chunk_text"
        df.drop(columns=["text"])
    else:
        text_column = "text"

    df = pd.DataFrame(rows)
    print(df.head())
    # 3. embeddings
    if config["preprocessing"]["embeddings"]["enabled"]:
        model_name = config["preprocessing"]["embeddings"]["model"]["name"]
        model = SentenceTransformer(model_name)
        df = add_embeddings(df, model, text_column=text_column)

    # 4. save
    store = DatasetStore(config["data"]["interim_path"])
    name = "embeddings" if config["preprocessing"]["embeddings"]["enabled"] else "chunked_256"
    save_path = store.save(df, name=name)
    print(f"Saved to: {save_path}")
    return df


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    df = run_pipeline(config)

    print("\nPipeline finished successfully")
    print(df.head())


if __name__ == "__main__":
    main()