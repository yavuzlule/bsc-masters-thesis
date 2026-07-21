import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.svm import SVC
from transformers import AutoTokenizer
from xgboost import XGBClassifier
import numpy as np
from typing import Union, List
from typing import Optional
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import torch
from tqdm import tqdm
import yaml
import json
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from safetensors.torch import load_file

from bsc_relish.utils.models import LanguageAwareClassifier



"""
ALL UTILS FUNCTIONS

1. INGESTION

input types: txt, folder of txts
output types: pandas Dataframe with columns: title, text, label

build_df_from_txt(directory_path, is_cooking, category): Reads all .txt files from the specified directory and constructs a DataFrame with columns for title, text, is_cooking, and category.
build_df_from_folder(directory_path, is_cooking, category): Reads all .txt files from the specified directory and constructs a DataFrame with columns for title, text, is_cooking, and category.

2. CLEANING

preprocess_text(text): Preprocesses the input text by lowercasing, normalizing line endings, removing multiple empty lines, collapsing repeated spaces/tabs, and removing strange OCR artifacts.

3. EMBEDDING

get_embedding(text, model, normalize=False, batch_size=32, show_progress_bar=False): Encodes a single string or list of strings into embeddings using the specified model.
get_embeddings_from_df(df, model, target_column='text', normalize=False, batch_size=32, show_progress_bar=True): Computes embeddings for a specified column in a DataFrame using the provided model.


4. TOKENIZATION

5. MODEL TRAINING

train_classical_model(model, X_train, y_train): Trains the specified classical model (e.g., Logistic Regression, SVM) on the training data.
train_transformer_model(device, model, data_loader, optimizer, scheduler, loss_fn): Trains a transformer model (e.g., BERT, RoBERTa) on the provided data loader.
train_language_aware_model(device, model, data_loader, optimizer, scheduler, loss_fn): Trains a language-aware model on the provided data loader.

6. MODEL SAVING

save_model(model, dir_name, model_name): Saves a trained model to disk with a custom name.

7. MODEL LOADING

load_model(dir_name, model_name): Loads a trained model from disk.

8. EVALUATION

evaluate_model(model, X_test, y_test): Evaluates the trained model on a test set and returns the accuracy and other metrics.
evaluate_logreg(model, X_test, y_test): Evaluates a Logistic Regression model on a test set and returns the accuracy and other metrics.
evaluate_svm(model, X_test, y_test): Evaluates a Support Vector Machine model on a test set and returns the accuracy and other metrics.
evaluate_xgboost(model, X_test, y_test): Evaluates an XGBoost model on a test set and returns the accuracy and other metrics.

9. OTHER UTILITIES

load_config(config_path): Loads a configuration file (e.g., JSON or YAML) and returns the configuration as a dictionary.
balance_classes(df, target_column): Balances the classes in a DataFrame based on the specified target column by oversampling the minority class or undersampling the majority class.

"""


# Tokenization

# Ingestion

def build_df_from_txt(file_path, is_cooking):
    df = pd.DataFrame(columns=["title", "text", "label"])

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        df = df.append({"title": Path(file_path).stem, "text": text, "label": is_cooking}, ignore_index=True)

    return df

def build_df_from_folder(directory_path, is_cooking):
    df = pd.DataFrame(columns=["title", "text", "label"])
    directory = Path(directory_path)

    for file_path in directory.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        df = df.append({"title": file_path.stem, "text": text, "label": is_cooking}, ignore_index=True)

    return df



def build_df_from_folder(directory_path, is_cooking):
    df = pd.DataFrame(columns=["title", "text", "label"])
    directory = Path(directory_path)

    for file_path in directory.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        df = df.append({
            "title": file_path.stem,
            "text": text,
            "label": is_cooking,
        })

    return df

# Embedding

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

# Cleaning

def preprocess_text(text):
    if text is None:
        return ""


    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove multiple empty lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # remove strange OCR artifacts (common in scanned books)
    text = re.sub(r"[^\w\s\n.,:;!?()\-/]", "", text)

    return text.strip()

# Evaluation

def evaluate_transformer_model(model, data_loader, device):
    model.eval()
    model.to(device)

    predictions = []
    actual_labels = []
    positive_probs = []

    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

            # Binary classification ROC-AUC
            positive_probs.extend(probs[:, 1].cpu().numpy())

    # Loss
    validation_loss = total_loss / len(data_loader)

    # Metrics
    validation_accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, zero_division=0)
    recall = recall_score(actual_labels, predictions, zero_division=0)
    f1 = f1_score(actual_labels, predictions, zero_division=0)

    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(actual_labels, positive_probs)

    cm = confusion_matrix(actual_labels, predictions)

    report = classification_report(
        actual_labels,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)
    report["loss"] = float(validation_loss)

    return report, cm

def evaluate_classical_model(model, X_test, y_test):
    total_loss = 0.0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    

    # Metrics
    validation_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    macro_f1 = f1_score(
        y_test, 
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, 
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)

    return report, cm

    
    total_loss = 0.0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    

    # Metrics
    validation_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    macro_f1 = f1_score(
        y_test, 
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, 
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)

    return report, cm

def evaluate_language_aware_model(model, data_loader, device):
    model.eval()
    model.to(device)

    predictions = []
    actual_labels = []
    positive_probs = []

    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=batch["lang_ids"].to(device)
            )

            logits = outputs

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

            # Binary classification ROC-AUC
            positive_probs.extend(probs[:, 1].cpu().numpy())

    # Loss
    validation_loss = total_loss / len(data_loader)

    # Metrics
    validation_accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, zero_division=0)
    recall = recall_score(actual_labels, predictions, zero_division=0)
    f1 = f1_score(actual_labels, predictions, zero_division=0)

    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(actual_labels, positive_probs)

    cm = confusion_matrix(actual_labels, predictions)

    report = classification_report(
        actual_labels,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)
    report["loss"] = float(validation_loss)

    return report, cm


# Model Training

def train_classical_model(model, dataset):
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

def train_transformer_model(device, model, data_loader, optimizer, scheduler, loss_fn):
    model = model.to(device)
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        assert labels.min() >= 0
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

def train_language_aware_model(device, model, data_loader, optimizer, scheduler, loss_fn):
    model = model.to(device)
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        optimizer.zero_grad()
        assert lang_ids.min() >= 0
        assert lang_ids.max() < model.lang_embedding.num_embeddings

        assert labels.min() >= 0
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lang_ids=lang_ids
        )

        logits = outputs
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

# Model Saving

def save_outputs_torch(
    run_dir, model, tokenizer, report, config,
    train_loss_arr, val_loss_arr, val_accuracy_arr,
    val_roc_auc_arr, val_f1_arr
):
    os.makedirs(run_dir, exist_ok=True)

    epochs_path = os.path.join(run_dir, "training_curves.npz")
    np.savez(
        epochs_path,
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
        accuracy=val_accuracy_arr,
        roc_auc=val_roc_auc_arr,
        f1_score=val_f1_arr
    )

    metrics_path = os.path.join(run_dir, "metrics.json")
    logs_path = os.path.join(run_dir, "logs.txt")
    config_path = os.path.join(run_dir, "config.yaml")

    if config.get("output", {}).get("save_model", False):
        save_path = os.path.join(run_dir, "model_weights.pth")
        torch.save(model.state_dict(), save_path)
        tokenizer.save_pretrained(run_dir)

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(logs_path, "w") as f:
        f.write(f"Validation Accuracy: {report.get('accuracy', 0):.4f}\n")
        f.write(f"Validation Loss: {report.get('loss', 0):.4f}\n")
        f.write(f"ROC AUC: {report.get('roc_auc', 0):.4f}\n")
        f.write(json.dumps(report, indent=2))

def save_outputs(run_dir, model, tokenizer, report, config, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr):

    #epochs_path = os.path.join(run_dir, "training_curves.npz")
    #np.savez(epochs_path, train_loss=train_loss_arr, val_loss=val_loss_arr, accuracy=val_accuracy_arr, roc_auc=val_roc_auc_arr, f1_score=val_f1_arr)

    metrics_path = os.path.join(run_dir, "metrics.json")
    logs_path = os.path.join(run_dir, "logs.txt")
    config_path = os.path.join(run_dir, "config.yaml")


    if config["output"]["save_model"]:
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

    # Predictions
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with open(logs_path, "w") as f:
        f.write(f"Validation Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Validation Loss: {report['loss']:.4f}\n")
        f.write(f"ROC AUC: {report['roc_auc']:.4f}\n")
        f.write(json.dumps(report, indent=2))

def save_epoch_outputs(run_dir, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr):
    epochs_path = os.path.join(run_dir, "training_curves.npz")
    np.savez(epochs_path, train_loss=train_loss_arr, val_loss=val_loss_arr, accuracy=val_accuracy_arr, roc_auc=val_roc_auc_arr, f1_score=val_f1_arr)

def save_model(model, dir_name, model_name):
    """
    Saves a trained model to disk with a custom name.
    """
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Model Loading

def load_model(model_path: str, params: dict):
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**params)

def load_model_safetensor(
    model_dir,
    base_model_name="bert-base-uncased",
):
    """
    model_dir/
    ├── model.safetensors
    ├── tokenizer files
    └── metrics.json
    """

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    state_dict = load_file(
        str(Path(model_dir) / "model.safetensors")
    )

    num_languages = state_dict[
        "lang_embedding.weight"
    ].shape[0]

    model = LanguageAwareClassifier(
        model_name=base_model_name,
        num_labels=2,
        num_languages=num_languages,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer, device

# Other Utilities

def balance_classes(df, label_col):
    if label_col not in df.columns:
        raise ValueError(f"Missing '{label_col}'. Columns: {df.columns.tolist()}")

    min_count = df[label_col].value_counts().min()

    print(f"Balancing classes to {min_count} samples each.")

    balanced_df = (
        df.groupby(label_col, group_keys=False)
          .sample(n=min_count, random_state=42)
          .reset_index(drop=True)
    )

    return balanced_df

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("Using CPU")
    return torch.device("cpu")
