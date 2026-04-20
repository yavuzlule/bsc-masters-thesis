import os
import torch
from transformers import BertTokenizer
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import logging
import yaml
from torch import nn


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
    


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #print(type(outputs))
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

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


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# -------------------------
# Main
# -------------------------
import os
from datetime import datetime


def main(df):
    config = load_config("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/bert_config.yaml")

    # Load data
    train_df = df
    target = config["data"]["target_column"]

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
    model = load_model(config["model"]["type"], config["model"]["params"])

    
    print("\n\n\nMODEL LOADED\n\n")

    """
    
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 128
    batch_size = 16
    num_epochs = 4  
    learning_rate = 2e-5
    """
    # Train
    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config['model']['params']['max_length'])
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config['model']['params']['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['params']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['model']['params']['batch_size'])

    # Evaluate
    accuracy, report = evaluate(model, val_dataloader, device="cpu")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

    # Save outputs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = config["model"]["name"]
    base_dir = config["output"]["base_dir"]

    run_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)




    model_path = os.path.join(run_dir, "model.joblib")
    metrics_path = os.path.join(run_dir, "metrics.json")
    logs_path = os.path.join(run_dir, "logs.txt")
    config_path = os.path.join(run_dir, "config.yaml")


    if config["output"]["save_model"]:
        joblib.dump(model, model_path)


    # Predictions
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    logging.basicConfig(
            filename=logs_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    main()