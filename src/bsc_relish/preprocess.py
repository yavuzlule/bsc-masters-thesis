import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml
from typing import Dict
from collections import defaultdict
from bsc_relish.embed import EmbeddingPipeline, add_embeddings_to_df, expand_embedding_column


# ---- Core file loading ---- #


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_txt_file(file_path: Path, encoding: str = "utf-8") -> str:
    """
    Load a single .txt file.

    Args:
        file_path: Path to file
        encoding: file encoding

    Returns:
        str: file content
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")


# ---- Directory traversal ---- #

def get_txt_files(root_dir: Path, recursive: bool = True) -> List[Path]:
    """
    Collect all .txt files from a directory.

    Args:
        root_dir: root folder
        recursive: whether to search subfolders

    Returns:
        List[Path]
    """
    if recursive:
        return list(root_dir.rglob("*.txt"))
    else:
        return list(root_dir.glob("*.txt"))


# ---- Main ingestion function ---- #

def txt_folder_to_df(
    root_dir: str,
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
    drop_empty: bool = True
) -> pd.DataFrame:
    """
    Convert a folder of .txt files into a DataFrame.

    Args:
        root_dir: path to root folder
        recursive: include subfolders
        encoding: file encoding
        drop_empty: remove empty texts

    Returns:
        pd.DataFrame
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    txt_files = get_txt_files(root_path, recursive=recursive)

    records = []

    for file_path in txt_files:
        text = load_txt_file(file_path, encoding=encoding)

        if drop_empty and not text.strip():
            continue

        records.append({
            "text": text,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "parent_folder": file_path.parent.name
        })

    df = pd.DataFrame(records)

    return df


from typing import List
import pandas as pd

def _chunk_text_by_words(text: str, max_words: int = 256) -> List[str]:
    """
    Split text into chunks of up to `max_words` words.
    """
    if max_words <= 0:
        raise ValueError("max_words must be > 0")

    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))

    return chunks


def split_text_into_chunks(
    df: pd.DataFrame,
    text_column: str = "text",
    *,
    max_words: int = 256
) -> pd.DataFrame:
    """
    Expand a DataFrame into word-chunk-level rows.

    Each row's text is split into chunks of at most `max_words` words.

    Adds:
        - chunk_text
        - chunk_index (position within original row)

    Returns:
        pd.DataFrame (one row per chunk)
    """

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    records = []

    for _, row in df.iterrows():
        text = str(row[text_column])
        chunks = _chunk_text_by_words(text, max_words=max_words)

        for idx, chunk in enumerate(chunks):
            record = row.to_dict()
            record["chunk_text"] = chunk
            record["chunk_index"] = idx
            records.append(record)

    return pd.DataFrame(records)


def txt_folder_to_df_with_labels(
    root_dir: str,
    label_map: Dict[str, int],
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
    drop_empty: bool = True,
    max_files: float = float('inf')
) -> pd.DataFrame:

    root_path = Path(root_dir)

    records = []
    folder_counts = defaultdict(int)  # track per-folder counts

    files = root_path.rglob("*.txt") if recursive else root_path.glob("*.txt")

    for file_path in files:
        folder_name = file_path.parent.name

        if folder_name not in label_map:
            continue

        # enforce per-folder limit
        if folder_counts[folder_name] >= max_files:
            continue

        text = file_path.read_text(encoding=encoding)

        if drop_empty and not text.strip():
            continue

        label = label_map[folder_name]

        records.append({
            "text": text,
            "label": label,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "parent_folder": folder_name
        })

        folder_counts[folder_name] += 1  # increment after adding

    return pd.DataFrame(records)

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



def run_preprocessing(config: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    txt -> df -> chunks -> embeddings
    """

    # ---- 1. Load raw text ---- #
    print("Constructing dataframes from txt files...")
    df = txt_folder_to_df_with_labels(
        root_dir="data/raw",
        label_map=config["data"]["label_map"],
        max_files=config["data"]["limit_files"]["max_files"] if config["data"]["limit_files"]["enabled"] else float('inf')
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





def main(config_path: str):
    config = load_config(config_path)

    # ---- Step 1: Preprocess ---- #
    if config["pipeline"]["run_preprocessing"]:
        print("Running preprocessing...")
        df = run_preprocessing(config)
        save_dataset(df, config)
    else:
        print("Skipping preprocessing...")
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