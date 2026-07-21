import os
from xml.parsers.expat import model
import mlflow
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml
import json
import joblib
import importlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from bsc_relish.sequence_classification.train.evaluate_la import evaluate
import logging
import yaml
from torch import device, nn
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
import torch.nn as nn

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


import torch
import torch.nn as nn
from transformers import AutoModel

class LanguageAwareClassifier(nn.Module):
    def __init__(self, model_name, num_labels, num_languages):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)

        hidden_size = self.encoder.config.hidden_size

        self.lang_embedding = nn.Embedding(num_languages, 32)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, lang_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lang_ids=lang_ids
        )

        text_repr = outputs.last_hidden_state[:, 0]  # CLS token
        lang_repr = self.lang_embedding(lang_ids)

        combined = torch.cat([text_repr, lang_repr], dim=1)

        return self.classifier(combined)
    
    
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



def train(device, model, data_loader, optimizer, scheduler, loss_fn):
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


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def balance_classes(df, label_col):
    min_count = df[label_col].value_counts().min()

    balanced_df = (
        df.groupby(label_col, group_keys=False)
          .sample(n=min_count, random_state=42)
          .reset_index(drop=True)
    )
    return balanced_df


def save_outputs(run_dir, model, tokenizer, report, config,
                 train_loss_arr, val_loss_arr, val_accuracy_arr,
                 val_roc_auc_arr, val_f1_arr):

    np.savez(
        os.path.join(run_dir, "training_curves.npz"),
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
        accuracy=val_accuracy_arr,
        roc_auc=val_roc_auc_arr,
        f1_score=val_f1_arr
    )

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join(run_dir, "logs.txt"), "w") as f:
        f.write(json.dumps(report, indent=2))

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def main(df):

    df = balance_classes(df, "label")
    df["language"] = df["language"].str.lower()
    df = df[df["language"].isin(language_order)]
    print(len(df))
    preprocess_config = load_config("/media/M2_disk/yavuz/bsc-masters-thesis/configs/preprocess.yaml")

    df["language"] = pd.Categorical(
        df["language"],
        categories=language_order,
        ordered=True,
    )
    
    UNK_ID = len(language_order)

    df["lang_id"] = df["language"].apply(
        lambda x: language_order.index(x) if x in language_order else UNK_ID
    )


    rows = df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")
    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        preprocess_config["preprocessing"]["chunking"]["max_words"],
    )

    df = pd.DataFrame(expanded)
    df["chunk_text"] = (
        "[LANG=" + df["language"].astype(str) + "] " + df["chunk_text"]
    )
    df = df[["chunk_text", "label", "language", "lang_id"]]

    texts = df["chunk_text"].reset_index(drop=True)
    labels = df["label"].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = config["model"]["params"]["num_epochs"]
    max_length = config["model"]["params"]["max_length"]
    batch_size = config["model"]["params"]["batch_size"]
    lr = float(config["model"]["params"]["learning_rate"])

    cv_cfg = config["training"]["cross_validation"]

    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"]
    )

    os.environ["MLFLOW_TRACKING_USERNAME"] = "yavuz"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "af>[9w?W}d]/:|xHx?N`hZv8{"
    # -------------------------
    # MLflow setup
    # -------------------------
    mlflow.set_tracking_uri("https://mlflow.dataviz.bsc.es")

    mlflow.set_experiment(f"{config['model']['name']} Text Classification")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    best_f1 = -1
    best_model_state = None
    best_fold_metrics = None

    # IMPORTANT: never hardcode credentials in code
    # use environment variables instead

    with mlflow.start_run(run_name=f"{config['model']['name']}-cv-{run_id}") as parent_run:

        # -------------------------
        # log global params once
        # -------------------------
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_length": max_length,
            "n_splits": cv_cfg["n_splits"],
        })

        fold_metrics = []

        # FIX: actually use skf splits
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):

            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):

                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]

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

                num_labels = len(df["label"].unique())
                num_languages = len(df["lang_id"].unique())

                model = LanguageAwareClassifier(
                    model_name=model_name,
                    num_labels=num_labels,
                    num_languages=train_df["lang_id"].nunique() + 1  # +1 for unknown language
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

                total_steps = len(train_loader) * epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(total_steps * 0.1),
                    num_training_steps=total_steps,
                )

                loss_fn = nn.CrossEntropyLoss()

                # -------------------------
                # training
                # -------------------------
                for epoch in range(epochs):
                    print(f"Epoch: {epoch}\n")
                    train(device, model, train_loader, optimizer, scheduler, loss_fn)

                # -------------------------
                # evaluation
                # -------------------------
                report, cm = evaluate(model, val_loader, device=device)

                # log per-fold metrics
                mlflow.log_metrics({
                    f"fold_macro_f1": report["macro_f1"],
                    f"fold_accuracy": report.get("accuracy", 0.0),
                })

                fold_metrics.append(report)

                if report["macro_f1"] > best_f1:
                    best_f1 = report["macro_f1"]
                    best_model_state = model.state_dict()
                    best_fold_metrics = report

        # -------------------------
        # aggregate metrics
        # -------------------------
        avg_report = {
            k: float(np.mean([f[k] for f in fold_metrics if isinstance(f[k], (int, float))]))
            for k in fold_metrics[0].keys()
            if isinstance(fold_metrics[0][k], (int, float))
        }

        mlflow.log_metrics({
            f"cv_avg_{k}": v for k, v in avg_report.items()
        })

        # -------------------------
        # save best model
        # -------------------------
        model.load_state_dict(best_model_state)

        run_dir = os.path.join(
            config["output"]["base_dir"],
            model_name,
            run_id
        )
        os.makedirs(run_dir, exist_ok=True)
        save_path = os.path.join(run_dir, "model_weights.pth")
        torch.save(model.state_dict(), save_path)
        tokenizer.save_pretrained(run_dir)

        # log as MLflow artifact (IMPORTANT)
        mlflow.log_artifacts(run_dir, artifact_path="model")

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({
                "best_fold": best_fold_metrics,
                "cv_average": avg_report
            }, f, indent=2)

        mlflow.log_metric("best_macro_f1", best_f1)


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_parquet(config["df"])

    main(df)

