import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Union, List
from typing import Optional
from pathlib import Path


# CLEAN THE DATA (txt file, df)
# DIVIDE IT BY TOKENS (construct df by dividing the txt file to chunks of max N tokens)
# COMPUTE EMBEDDING (string, df)

def get_embedding(
    text: Union[str, List[str]],
    model,
    *,
    normalize: bool = False,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Encode a single string or list of strings into embeddings.

    Args:
        text: str or list of str
        model: SentenceTransformer-like model
        normalize: whether to L2-normalize embeddings
        batch_size: batch size for encoding
        show_progress_bar: display progress bar for large inputs

    Returns:
        np.ndarray: embedding vector(s)
    """
    embeddings = model.encode(
        text,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize
    )
    return embeddings

def get_embeddings_from_df(
    df: pd.DataFrame,
    model,
    target_column: Optional[str] = "text",
    *,
    normalize: bool = False,
    batch_size: int = 32,
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a DataFrame column.

    Args:
        df: pandas DataFrame
        model: SentenceTransformer-like model
        target_column: column containing text (default: 'text')
        normalize: whether to L2-normalize embeddings
        batch_size: batch size for encoding
        show_progress_bar: display progress bar

    Returns:
        np.ndarray: embeddings
    """
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")

    texts = df[target_column].astype(str).tolist()

    return get_embedding(
        texts,
        model,
        normalize=normalize,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar
    )

def train_model(model, dataset):
    """
    model: sklearn model (LogisticRegression, SVM with probability, XGBoost, etc.)
    dataset: pandas DataFrame with 'text' and 'label'
    Returns trained model
    """
    X = dataset['mini_lm_embeddings'].tolist()  # embeddings must be precomputed and added to df
    y = dataset['is_cooking'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model training completed. Test accuracy: {model.score(X_test, y_test):.4f}")
    return model



def build_df_from_txt(directory_path, is_cooking, category):
    rows = []
    directory = Path(directory_path)

    for file_path in directory.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        rows.append({
            "title": file_path.stem,
            "text": text,
            "is_cooking": is_cooking,
            "category": category
        })
 
    return pd.DataFrame(rows)


def preprocess_text(text):
    if text is None:
        return ""

    # lowercase
    text = text.lower()

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove multiple empty lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # remove strange OCR artifacts (common in scanned books)
    text = re.sub(r"[^\w\s\n.,:;!?()\-/]", "", text)

    return text.strip()

def save_model(model, dir_name, model_name):
    """
    Saves a trained model to disk with a custom name.
    """
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    

def divide_books_on_chars(df, subdivision_size):
    rows = []

    for row in df.itertuples(index=False):
        title = row.title
        text = row.text

        for i in range(0, len(text), subdivision_size):
            chunk = text[i:i + subdivision_size]
            index_of_subtext = i // subdivision_size + 1

            rows.append({
                "title": title,
                "index_of_subtext": index_of_subtext,
                "text": chunk
            })

    return pd.DataFrame(rows)