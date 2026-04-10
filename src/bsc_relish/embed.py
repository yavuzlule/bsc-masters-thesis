"""
Embedding pipeline module.

Usage (as import):
    from embedding_pipeline import EmbeddingPipeline

    pipeline = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
    embeddings = pipeline.encode_dataframe(df)

Usage (CLI-style):
    python embedding_pipeline.py --input data.csv --output embeddings.npy
"""

from typing import Optional
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---- Core reusable functions ---- #

def get_embedding(
    text,
    model,
    *,
    normalize: bool = False,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> np.ndarray:
    return model.encode(
        text,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize
    )


def add_embeddings_to_df(
    df: pd.DataFrame,
    model,
    target_column: str = "text",
    *,
    embedding_column: str = "embedding",
    model_column: str = "embedding_model",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress_bar: bool = True
) -> pd.DataFrame:
    """
    Returns a copy of df with embedding + model name columns added.
    """

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")

    texts = df[target_column].astype(str).tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize
    )

    df_out = df.copy()

    # Store embeddings (object dtype: list/array per row)
    df_out[embedding_column] = list(embeddings)

    # Store model name for traceability
    model_name = getattr(model, "name_or_path", "unknown_model")
    df_out[model_column] = model_name

    return df_out


def expand_embedding_column(
    df: pd.DataFrame,
    embedding_column: str = "embedding",
    prefix: str = "emb"
) -> pd.DataFrame:
    """
    Expand a column of numpy arrays into multiple numeric columns.

    Args:
        df: input DataFrame
        embedding_column: column containing np.array embeddings
        prefix: prefix for generated columns

    Returns:
        pd.DataFrame with flattened embedding features
    """

    if embedding_column not in df.columns:
        raise ValueError(f"Column '{embedding_column}' not found")

    embeddings = np.vstack(df[embedding_column].values)

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"{prefix}_{i}" for i in range(embeddings.shape[1])]
    )

    df_out = df.drop(columns=[embedding_column]).reset_index(drop=True)
    emb_df = emb_df.reset_index(drop=True)

    return pd.concat([df_out, emb_df], axis=1)
# ---- Pipeline class (recommended pattern) ---- #

class EmbeddingPipeline:
    """
    High-level interface for embedding generation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_text(self, text, **kwargs) -> np.ndarray:
        return get_embedding(text, self.model, **kwargs)

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str = "text",
        **kwargs
    ) -> np.ndarray:
        return add_embeddings_to_df(
            df,
            self.model,
            target_column=target_column,
            **kwargs
        )


# ---- CLI entry point ---- #

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from a DataFrame")

    parser.add_argument("--input", type=str, required=True, help="Input dataframe file (parquet)")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    parser.add_argument("--column", type=str, default="text", help="Text column name")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Model name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Initialize pipeline
    pipeline = EmbeddingPipeline(model_name=args.model)

    # Compute embeddings
    embeddings = pipeline.encode_dataframe(
        df,
        target_column=args.column,
        batch_size=args.batch_size,
        normalize=args.normalize,
        show_progress_bar=True
    )

    # Save output
    np.save(args.output, embeddings)

    print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()