import os
from xml.parsers.expat import model
import mlflow
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, XLMRobertaForSequenceClassification
import yaml
import json
import joblib
import importlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from bsc_relish.sequence_classification.train.evaluate import evaluate
import logging
import yaml
from torch import device, nn
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

import torch.nn as nn

#from bsc_relish.visualize_report import confusion_matrix_heatmap

# Standard approach: let BERT handle it

# During training:
# - Forward pass outputs logits
# - Loss function (CrossEntropyLoss) applies softmax internally
# - You get probabilities during inference with softmax

language_order = ['english', 
                  'old german', 
                  'catalan', 
                  'multiple', 
                  'italian', 
                  'latin',
                  'french', 
                  'old french', 
                  'venetian/italian', 
                  'old danish', 
                  'middle dutch', 
                  'middle low german', 
                  'early modern english', 
                  'middle french', 
                  'spanish', 
                  'german', 
                  'dutch']



class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, lang_ids, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.lang_ids = lang_ids.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        lang_id = self.lang_ids.iloc[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long),
        }


def train(device, model, data_loader, optimizer, scheduler,
          loss_fn):
    model = model.to(device)
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        # ensure dtype safety (important for many loss functions like CrossEntropyLoss)
        loss = loss_fn(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

# -------------------------
# Utils
# -------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, params: dict):
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**params)


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


def save_outputs(run_dir, model, tokenizer, report, config, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr):


    epochs_path = os.path.join(run_dir, "training_curves.npz")
    np.savez(epochs_path, train_loss=train_loss_arr, val_loss=val_loss_arr, accuracy=val_accuracy_arr, roc_auc=val_roc_auc_arr, f1_score=val_f1_arr)

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



import os
import copy
import numpy as np
import torch
import mlflow
import pandas as pd

from datetime import datetime
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold


def build_model(config):
    """
    Replace with your actual model constructor.
    """
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["params"]["num_labels"]
    )


def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def setup_optimizer_scheduler(model, config, total_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["params"]["learning_rate"])
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    return evaluate(model, loader, device)  # your existing function


def run_single_training(model, train_loader, val_loader, config, device, mlflow_prefix=""):
    epochs = config["model"]["params"]["num_epochs"]

    optimizer, scheduler = setup_optimizer_scheduler(
        model,
        config,
        total_steps=len(train_loader) * epochs
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device
        )

        report, _ = evaluate_model(model, val_loader, device)
        f1 = report["macro_f1"]

        mlflow.log_metric(f"{mlflow_prefix}train_loss", train_loss, step=epoch)
        mlflow.log_metric(f"{mlflow_prefix}val_f1", f1, step=epoch)

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_f1

import os
import json
import numpy as np
import pandas as pd
import torch
import mlflow

from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

def main(df, config_file):

    # -----------------------
    # Config
    # -----------------------

    os.environ["MLFLOW_TRACKING_USERNAME"] = "yavuz"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "af>[9w?W}d]/:|xHx?N`hZv8{"
    config = config_file

    preprocess_config = load_config(
        "/media/M2_disk/yavuz/bsc-masters-thesis/configs/preprocess.yaml"
    )

    model_name = config["model"]["name"]
    batch_size = config["model"]["params"]["batch_size"]
    max_length = config["model"]["params"]["max_length"]
    epochs = config["model"]["params"]["num_epochs"]

    base_dir = config["output"]["base_dir"]

    cross_validation = config["training"]["cross_validation"]["enabled"]

    # -----------------------
    # Run ID + output dir (FIXED ORDER)
    # -----------------------
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # -----------------------
    # Preprocess
    # -----------------------
    df = balance_classes(df, "label")
    df["language"] = df["language"].str.lower()

    df["language"] = pd.Categorical(
        df["language"],
        categories=language_order,
        ordered=True,
    )
    df["lang_id"] = df["language"].cat.codes

    rows = df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        preprocess_config["preprocessing"]["chunking"]["max_words"],
    )

    df = pd.DataFrame(expanded)
    df = df[["chunk_text", "label", "language", "lang_id"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Common dataset (for CV)
    # -----------------------
    full_dataset = TextClassificationDataset(
        df["chunk_text"],
        df["label"],
        df["lang_id"],
        tokenizer,
        max_length,
    )

    labels = df["label"].values

    # ============================================================
    # SIMPLE TRAIN/VAL SPLIT MODE
    # ============================================================
    if not cross_validation:

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )

        train_dataset = TextClassificationDataset(
            train_df["chunk_text"],
            train_df["label"],
            train_df["lang_id"],
            tokenizer,
            max_length,
        )

        val_dataset = TextClassificationDataset(
            val_df["chunk_text"],
            val_df["label"],
            val_df["lang_id"],
            tokenizer,
            max_length,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(df["label"].unique()),
            use_safetensors=True,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["params"]["learning_rate"]),
        )

        total_steps = len(train_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        loss_fn = nn.CrossEntropyLoss()

        mlflow.set_tracking_uri("https://mlflow.dataviz.bsc.es")
        mlflow.set_experiment("XLMRoBERTa Text Classification")

        best_f1 = 0.0

        with mlflow.start_run(run_name=f"xlmroberta-{run_id}"):

            mlflow.log_params({
                "learning_rate": config["model"]["params"]["learning_rate"],
                "batch_size": batch_size,
                "num_epochs": epochs,
                "model_name": model_name,
                "max_length": max_length,
                "training_size": len(train_dataset),
                "validation_size": len(val_dataset),
            })

            for epoch in range(epochs):

                train_loss = train(
                    device,
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                )

                report, cm = evaluate(model, val_loader, device=device)

                f1 = report["macro_f1"]

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_f1", f1, step=epoch)
                mlflow.log_metric("val_loss", report["loss"], step=epoch)

                if f1 > best_f1:
                    best_f1 = f1

                    best_dir = os.path.join(run_dir, "best_model")
                    os.makedirs(best_dir, exist_ok=True)

                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

                    torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))

            save_outputs(
                run_dir,
                model,
                tokenizer,
                report,
                config,
                None,
                None,
                None,
                None,
                None,
            )

        return best_f1

    # ============================================================
    # CROSS VALIDATION MODE
    # ============================================================
    else:

        skf = StratifiedKFold(
            n_splits=config["training"]["cross_validation"]["n_splits"],
            shuffle=True,
            random_state=42,
        )

        fold_scores = []

        mlflow.set_tracking_uri("https://mlflow.dataviz.bsc.es")
        mlflow.set_experiment("XLMRoBERTa Text Classification")

        with mlflow.start_run(run_name=f"cross_validation-{run_id}"):

            for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):

                with mlflow.start_run(run_name=f"fold_{fold}", nested=True):

                    train_subset = Subset(full_dataset, train_idx)
                    val_subset = Subset(full_dataset, val_idx)

                    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=batch_size)

                    model_fold = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=len(df["label"].unique()),
                        use_safetensors=True,
                    ).to(device)

                    optimizer = torch.optim.AdamW(
                        model_fold.parameters(),
                        lr=float(config["model"]["params"]["learning_rate"]),
                    )

                    total_steps = len(train_loader) * epochs

                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(total_steps * 0.1),
                        num_training_steps=total_steps,
                    )

                    loss_fn = nn.CrossEntropyLoss()

                    best_f1 = 0.0
                    best_state = None

                    for epoch in range(epochs):

                        train_loss = train(
                            device,
                            model_fold,
                            train_loader,
                            optimizer,
                            scheduler,
                            loss_fn,
                        )

                        report, cm = evaluate(
                            model_fold,
                            val_loader,
                            device=device,
                        )

                        f1 = report["macro_f1"]

                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_f1", f1, step=epoch)
                        mlflow.log_metric("val_loss", report["loss"], step=epoch)
                        mlflow.log_metric("val_accuracy", report["accuracy"], step=epoch)

                        if f1 > best_f1:
                            best_f1 = f1
                            best_state = model_fold.state_dict()

                            fold_dir = os.path.join(run_dir, f"fold_{fold}_best")
                            os.makedirs(fold_dir, exist_ok=True)

                            model_fold.save_pretrained(fold_dir)
                            tokenizer.save_pretrained(fold_dir)
                            torch.save(best_state, os.path.join(fold_dir, "pytorch_model.bin"))

                    fold_scores.append(best_f1)

        return float(np.mean(fold_scores))
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    df = pd.read_parquet(config["df"])

    main(df, config)

