from pathlib import Path
from typing import List, Dict


def load_txt_files(
    root_dir: str,
    encoding: str = "utf-8",
    recursive: bool = True,
    drop_empty: bool = True,
) -> List[Dict]:

    root = Path(root_dir)
    files = root.rglob("*.txt") if recursive else root.glob("*.txt")

    records = []

    for f in files:
        text = f.read_text(encoding=encoding, errors="ignore")

        if drop_empty and not text.strip():
            continue

        records.append({
            "text": text,
            "file_name": f.name,
            "file_path": str(f),
            "parent_folder": f.parent.name
        })

    return records