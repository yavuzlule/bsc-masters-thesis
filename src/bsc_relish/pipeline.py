from datetime import datetime
import argparse
from pathlib import Path
import pandas as pd
from bsc_relish.preprocess import (
    run_preprocessing,
    save_dataset,
    load_config
)
from bsc_relish.embed import EmbeddingPipeline, add_embeddings_to_df, expand_embedding_column
from bsc_relish.train_logreg import main as train_main


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

        
        print("\nRunning training...")
        if config["pipeline"]["run_preprocessing"]:
            train_main(df)
        else:
            interim_path = Path(config["data"]["interim_path"])

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