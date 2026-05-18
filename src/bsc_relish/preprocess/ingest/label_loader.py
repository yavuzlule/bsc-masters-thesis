from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_labeled_txt_files(
    root_dir: str,
    label_map: Dict[str, int],
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
    max_files: int = 10**9
) -> List[Dict]:

    root = Path(root_dir)
    files = root.rglob("*.txt") if recursive else root.glob("*.txt")

    records = []
    folder_counts = defaultdict(int)

    for f in files:
        folder = f.parent.name

        if folder not in label_map:
            continue

        if folder_counts[folder] >= max_files:
            continue

        text = f.read_text(encoding=encoding, errors="ignore")

        if not text.strip():
            continue

        records.append({
            "text": text,
            "label": label_map[folder],
            "file_name": f.name,
            "file_path": str(f),
            "parent_folder": folder
        })

        folder_counts[folder] += 1

    return records