from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import os


# -----------------------------
# Canonical intermediate type
# -----------------------------

@dataclass
class Document:
    """
    Canonical representation of any text input.
    This is the ONLY output of the ingestion layer.
    """
    text: str
    metadata: Dict[str, Any]


# -----------------------------
# Main loader
# -----------------------------

class FileLoader:
    """
    Handles multiple raw input formats and normalizes them into Documents.

    Supported inputs:
    - folder of .txt files
    - single .txt file
    - raw string
    - list of strings
    - list of file paths
    """

    def load(self, source: Union[str, Path, List[str], List[Path]]) -> List[Document]:
        if isinstance(source, list):
            return self._load_list(source)

        if isinstance(source, (str, Path)):
            source = str(source)

            if os.path.isdir(source):
                return self._load_folder(Path(source))

            if os.path.isfile(source):
                return self._load_file(Path(source))

            # otherwise treat as raw text
            return self._load_string(source)

        raise TypeError(f"Unsupported input type: {type(source)}")

    # -----------------------------
    # Handlers
    # -----------------------------

    def _load_folder(self, folder_path: Path) -> List[Document]:
        documents = []

        for file_path in sorted(folder_path.rglob("*.txt")):
            documents.append(self._load_file(file_path)[0])

        return documents

    def _load_file(self, file_path: Path) -> List[Document]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        return [
            Document(
                text=text,
                metadata={
                    "source_type": "file",
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                },
            )
        ]

    def _load_string(self, text: str) -> List[Document]:
        return [
            Document(
                text=text,
                metadata={
                    "source_type": "string",
                    "length": len(text),
                },
            )
        ]

    def _load_list(self, items: List[Union[str, Path]]) -> List[Document]:
        documents = []

        for item in items:
            if isinstance(item, Path) or (isinstance(item, str) and os.path.exists(item)):
                documents.extend(self._load_file(Path(item)))
            else:
                documents.extend(self._load_string(str(item)))

        return documents