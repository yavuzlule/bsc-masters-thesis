from datetime import datetime
import json
import os
import argparse
from pathlib import Path
import yaml
import pandas as pd

# Your modules
from bsc_relish.preprocess import (
    txt_folder_to_df,
    txt_folder_to_df_with_labels,
    split_text_into_chunks
)
from bsc_relish.embed import EmbeddingPipeline, add_embeddings_to_df, expand_embedding_column

# Import train logic
from bsc_relish.train import main as train_main


# -------------------------
# Config
# -------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Pipeline steps
# -------------------------

def run_preprocessing(config: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    txt -> df -> chunks -> embeddings
    """

    # ---- 1. Load raw text ---- #
    print("Constructing dataframes from txt files...")
    df = txt_folder_to_df_with_labels(
        root_dir="data/raw",
        label_map={
            "archive-americana-cookbook-download-desc": 1,
            "archive-americana-download-desc": 1,
            "archive-americana-recipe-download-desc": 1,
            "archive-americana-fiction-literature-technology-download-desc": 0
        },
        max_files=100
    )
    print("Constructing dataframes from txt files...")

    # ---- 2. Chunking ---- #
    if config["preprocessing"]["chunking"]["enabled"]:
        df = split_text_into_chunks(
            df,
            text_column="text",
            max_words=config["preprocessing"]["chunking"]["max_words"],
        )
        text_column = "chunk_text"
    else:
        text_column = "text"

    # ---- 3. Embeddings ---- #
    if config["preprocessing"]["embeddings"]["enabled"]:
        pipeline = EmbeddingPipeline(
            model_name=config["preprocessing"]["embeddings"]["model_name"]
        )

        df = add_embeddings_to_df(
            df,
            pipeline.model,
            target_column=text_column,
            embedding_column="embedding",
            model_column="embedding_model",
        )
        df = expand_embedding_column(df)

    return df


def save_dataset(df: pd.DataFrame, config: dict):
    base_dir = "data/interim"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(base_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    dataset_path = os.path.join(save_dir, "dataset.parquet")
    metadata_path = os.path.join(save_dir, "metadata.json")

    # Save dataset
    df.to_parquet(dataset_path, index=False, compression="zstd")

    # Save metadata (very useful later)
    metadata = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "target": config["data"]["target_column"],
        "example_row": df.sample(1, random_state=42).iloc[0].to_dict()
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {dataset_path}")



# -------------------------
# Main pipeline
# -------------------------

def main(config_path: str):
    config = load_config(config_path)

    # ---- Step 1: Preprocess ---- #
    if config["pipeline"]["run_preprocessing"]:
        print("Running preprocessing...")
        df = run_preprocessing(config)
        save_dataset(df, config)
    else:
        print("Skipping preprocessing...")

    # ---- Step 2: Training ---- #
    if config["pipeline"]["run_training"]:

        
        print("Running training...")
        if config["pipeline"]["run_preprocessing"]:
            train_main(df)
        else:
            interim_path = Path(config["data"]["train_path"])

            # get all timestamped directories
            dirs = [d for d in interim_path.iterdir() if d.is_dir()]

            if not dirs:
                raise FileNotFoundError(f"No subdirectories found in {interim_path}")

            # latest by lexicographic order (safe due to YYYY-MM-DD_HH-MM-SS format)
            latest_dir = max(dirs, key=lambda d: d.name)

            dataset_path = latest_dir / "dataset.parquet"

            if not dataset_path.exists():
                raise FileNotFoundError(f"No dataset.parquet in {latest_dir}")

            df = pd.read_parquet(dataset_path)
            train_main(df)
    else:
        print("Skipping training...")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full ML pipeline")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    main(args.config)