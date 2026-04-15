from pathlib import Path
from typing import List, Optional
import pandas as pd


# ---- Core file loading ---- #

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

from pathlib import Path
import pandas as pd
from typing import Dict

from pathlib import Path
from typing import Dict
import pandas as pd
from collections import defaultdict

def txt_folder_to_df_with_labels(
    root_dir: str,
    label_map: Dict[str, int],
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
    drop_empty: bool = True,
    max_files: int = 5
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