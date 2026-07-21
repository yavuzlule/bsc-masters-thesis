from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer, DistilBertTokenizer

def chunk_by_tokens(config, tokenizer, text: str, max_tokens: int = 256, stride: int = 64) -> List[str]:
    """
    Splits text into chunks based on tokenizer subword tokens.
    stride: optional overlap between chunks (useful for retrieval tasks)
    """

    tokens = tokenizer.encode(text)

    chunks = []
    step = max_tokens - stride if max_tokens > stride else max_tokens
    for i in range(0, len(tokens), step):
        chunk_ids = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


def expand_chunks(config, tokenizer, rows: List[Dict], max_tokens: int = 256, stride: int = 64) -> List[Dict]:
    expanded = []

    for row in tqdm(rows, desc="Expanding token chunks", unit="row"):
        chunks = chunk_by_tokens(config=config, tokenizer=tokenizer, text=row["text"], max_tokens=max_tokens, stride=stride)

        for i, chunk in enumerate(chunks):
            new_row = dict(row)
            new_row["chunk_text"] = chunk
            new_row["chunk_index"] = i
            expanded.append(new_row)

    return expanded