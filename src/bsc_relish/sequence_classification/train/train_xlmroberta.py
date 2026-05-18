import os
from xml.parsers.expat import model
import mlflow
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
import yaml
import json
import joblib
import importlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from bsc_relish.evaluate import evaluate
import logging
import yaml
from torch import device, nn
from transformers import get_linear_schedule_with_warmup


from transformers import BertForSequenceClassification
import torch.nn as nn

from bsc_relish.visualize_report import confusion_matrix_heatmap

# Standard approach: let BERT handle it

# During training:
# - Forward pass outputs logits
# - Loss function (CrossEntropyLoss) applies softmax internally
# - You get probabilities during inference with softmax



class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
def train(device, model, data_loader, optimizer, #scheduler, 
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
        #scheduler.step()

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



# -------------------------
# Main
# -------------------------
import os
from datetime import datetime




def main(df):
    config = load_config("configs/xlmroberta_config.yaml")

    # Load data
    target = config["data"]["target_column"]
    df = balance_classes(df, target)
    texts = df['chunk_text']
    labels = df['label']

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42
    )

    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    

    # Build pipeline
    #preprocessor = build_preprocessor(config)
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = XLMRobertaForSequenceClassification.from_pretrained(config["model"]["name"], num_labels=2, use_safetensors=True)
    print("\n\n\nMODEL LOADED\n\n")

    # Train
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"],  use_safetensors=True)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config['model']['params']['max_length'])
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config['model']['params']['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['params']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['model']['params']['batch_size'])
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['model']['params']['learning_rate']))

    epochs = config['model']['params']['num_epochs']
    total_steps = len(train_dataloader) * epochs
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    """

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels.values
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)


    val_loss_arr = np.zeros(epochs)
    train_loss_arr = np.zeros(epochs)
    val_accuracy_arr = np.zeros(epochs)
    val_roc_auc_arr = np.zeros(epochs)
    val_f1_arr = np.zeros(epochs)


    best_f1 = -1
    start = datetime.now()
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "yavuz"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "af>[9w?W}d]/:|xHx?N`hZv8{"

    mlflow.set_tracking_uri("https://mlflow.dataviz.bsc.es")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment("XLMRoBERTa Text Classification")
    mlflow.start_run(run_name=f"xlmroberta-{run_id}")
    
    mlflow.log_param("learning_rate", config["model"]["params"]["learning_rate"])
    mlflow.log_param("batch_size", config["model"]["params"]["batch_size"])
    mlflow.log_param("num_epochs", config["model"]["params"]["num_epochs"])
    mlflow.log_param("model_name", config["model"]["name"])
    mlflow.log_param("max_length", config["model"]["params"]["max_length"])
    mlflow.log_param("training_data_size", len(train_dataset))
    mlflow.log_param("validation_data_size", len(val_dataset))
    
    for epoch in range(epochs):
        # ---- Train ----
        train_loss = train(
            device,
            model,
            train_dataloader,
            optimizer,
            #scheduler,
            loss_fn
        )

        # ---- Validation ----
        report, cm = evaluate(
            model,
            val_dataloader,  # validation dataloader required
            device=device
        )

        f1 = report["macro avg"]["f1-score"]
        mlflow.log_metric("val_f1", f1, step=epoch)
        mlflow.log_metric("val_accuracy", report["accuracy"], step=epoch)
        mlflow.log_metric("val_loss", report["loss"], step=epoch)
        mlflow.log_metric("val_roc_auc", report["roc_auc"], step=epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        if f1 > best_f1:
            best_f1 = f1
            mlflow.log_metric("best_f1", best_f1)

        val_accuracy_arr[epoch] = report["accuracy"]
        val_loss_arr[epoch] = report["loss"]
        train_loss_arr[epoch] = train_loss
        val_roc_auc_arr[epoch] = report["roc_auc"]
        val_f1_arr[epoch] = report["macro avg"]["f1-score"]


        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Accuracy: {report['accuracy']:.4f}"
        )


    end = datetime.now()
    hours, remainder = divmod((end - start).total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")


    report, cm = evaluate(model, val_dataloader, device=device)
  
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Loss: {report['loss']:.4f}")
    print(f"ROC AUC: {report['roc_auc']:.4f}")
    print(report)



    model_name = config["model"]["name"]
    base_dir = config["output"]["base_dir"]

    run_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)


    save_outputs(run_dir, model, tokenizer, report, config, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr)

    np.save(os.path.join(run_dir, "confusion_matrix.npy"), cm)
    np.savez(os.path.join(run_dir, "training_curves.npz"), train_loss=train_loss_arr, val_loss=val_loss_arr, accuracy=val_accuracy_arr, roc_auc=val_roc_auc_arr, f1_score=val_f1_arr)   
    #visualize_report(run_dir)
    confusion_matrix_heatmap(run_dir)


    if config["output"]["save_model"]:
        mlflow.log_artifacts(os.path.join(run_dir))

    mlflow.end_run()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    df = pd.read_parquet(config["df"])

    main(df)