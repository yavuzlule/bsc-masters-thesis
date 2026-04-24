import os
from xml.parsers.expat import model
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
from torch import device, nn
from transformers import get_linear_schedule_with_warmup


from transformers import BertForSequenceClassification
import torch.nn as nn

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
    
def train(model, data_loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0

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

        loss = loss_fn(outputs.logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits  # Shape: (batch_size, 2)
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            food_probs = probs[:, 1]  # Probability of class 1 (food)
            preds = (food_probs > 0.5).long()  # Now threshold makes sense

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)





# -------------------------
# Main
# -------------------------
import os
from datetime import datetime




def main(df):
    config = load_config("configs/bert_config.yaml")

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
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2  # binary classification
    )
    
    print("\n\n\nMODEL LOADED\n\n")

    # Train
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["name"])
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config['model']['params']['max_length'])
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, config['model']['params']['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['params']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['model']['params']['batch_size'])
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['model']['params']['learning_rate']))

    epochs = config['model']['params']['num_epochs']
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )


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
    for epoch in range(epochs):
        loss = train(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            loss_fn
        )
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")



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




    metrics_path = os.path.join(run_dir, "metrics.json")
    logs_path = os.path.join(run_dir, "logs.txt")
    config_path = os.path.join(run_dir, "config.yaml")


    if config["output"]["save_model"]:
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

    # Predictions
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=0)

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
    config = load_config("configs/bert_config.yaml")
    df = pd.read_parquet(config["df"])

    main(df)