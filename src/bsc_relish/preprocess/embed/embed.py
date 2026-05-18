import pandas as pd
from typing import Any


def add_embeddings(
    df: pd.DataFrame,
    model: Any,
    text_column: str = "text",
    embedding_column: str = "embedding",
) -> pd.DataFrame:


    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    df = df.copy()
    df[embedding_column] = list(embeddings)

    return df