from typing import List, Dict


def chunk_text(text: str, max_words: int = 256) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def expand_chunks(rows: List[Dict], max_words: int = 256) -> List[Dict]:
    expanded = []

    for row in rows:
        chunks = chunk_text(row["text"], max_words=max_words)

        for i, chunk in enumerate(chunks):
            new_row = dict(row)
            new_row["chunk_text"] = chunk
            new_row["chunk_index"] = i
            expanded.append(new_row)

    return expanded