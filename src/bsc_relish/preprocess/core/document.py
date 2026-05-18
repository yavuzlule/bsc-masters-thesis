from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]