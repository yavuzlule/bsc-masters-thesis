from dataclasses import dataclass
from typing import List, Union, Dict, Any


@dataclass
class Document:
    """
    Canonical document format used across the pipeline.
    """
    text: str
    metadata: Dict[str, Any]


class TextLoader:
    """
    Handles in-memory text inputs only.

    Responsibilities:
    - raw string → Document
    - list of strings → Documents

    Does NOT:
    - read files
    - traverse directories
    - chunk or clean text
    """

    def load(self, input_data: Union[str, List[str]]) -> List[Document]:
        if isinstance(input_data, str):
            return self._load_single(input_data)

        if isinstance(input_data, list):
            return self._load_list(input_data)

        raise TypeError(f"Unsupported type: {type(input_data)}")

    def _load_single(self, text: str) -> List[Document]:
        return [
            Document(
                text=text,
                metadata={
                    "source_type": "string",
                    "length": len(text),
                },
            )
        ]

    def _load_list(self, texts: List[str]) -> List[Document]:
        return [
            Document(
                text=t,
                metadata={
                    "source_type": "string_list",
                    "index": i,
                    "length": len(t),
                },
            )
            for i, t in enumerate(texts)
        ]