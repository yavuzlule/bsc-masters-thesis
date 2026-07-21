import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import get_linear_schedule_with_warmup
import mlflow
import torch.nn as nn
from datetime import datetime
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from bsc_relish.utils.models import TextClassificationDataset
from bsc_relish.utils.utils import balance_classes, load_config, evaluate_transformer_model, save_outputs, train_transformer_model


def main(df, config):
    config = config
    preprocess_config = load_config("configs/preprocess.yaml")

    df = balance_classes(df, "label")
    print(len(df))
 

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


    # Reset indices
    df = df.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = config["model"]["params"]["num_epochs"]
    max_length = config["model"]["params"]["max_length"]
    batch_size = config["model"]["params"]["batch_size"]
    lr = float(config["model"]["params"]["learning_rate"])

    # Train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],  # remove if labels are too sparse
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Train dataset
    train_dataset = TextClassificationDataset(
        train_df["chunk_text"],
        train_df["label"],
        tokenizer,
        max_length,
    )

    # Validation dataset
    val_dataset = TextClassificationDataset(
        val_df["chunk_text"],
        val_df["label"],
        tokenizer,
        max_length,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['params']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['model']['params']['batch_size'])
    
    # Print class distribution after balancing
    print("\nClass distribution after balancing:")
    print(df["label"].value_counts())
    
    

    # Build pipeline
    num_labels = len(df["label"].unique())
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        use_safetensors=True,
    )


    print(f"Using device: {device}")
    print(f"Model: {config['model']['name']} with {config['model']['params']['num_epochs']} epochs, batch size {config['model']['params']['batch_size']}, learning rate {config['model']['params']['learning_rate']} \n")
   
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['model']['params']['learning_rate']))


    epochs = config['model']['params']['num_epochs']
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()


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
    mlflow.set_experiment(f"{config['model']['name']} Text Classification")
    mlflow.start_run(run_name=f"{config['model']['name']}-{run_id}")
    
    mlflow.log_param("learning_rate", config["model"]["params"]["learning_rate"])
    mlflow.log_param("batch_size", config["model"]["params"]["batch_size"])
    mlflow.log_param("num_epochs", config["model"]["params"]["num_epochs"])
    mlflow.log_param("model_name", config["model"]["name"])
    mlflow.log_param("max_length", config["model"]["params"]["max_length"])
    mlflow.log_param("training_data_size", len(train_dataset))
    mlflow.log_param("validation_data_size", len(val_dataset))


    for epoch in range(epochs):

        
        # ---- Train ----
        train_loss = train_transformer_model(
            device,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            loss_fn
        )

        # ---- Validation ----
        report, cm = evaluate_transformer_model(
            model,
            val_dataloader,  # validation dataloader required
            device=device
        )

        f1 = report["macro_f1"]
        mlflow.log_metric("val_accuracy", report["accuracy"], step=epoch)
        mlflow.log_metric("val_precision", report["precision"], step=epoch)
        mlflow.log_metric("val_recall", report["recall"], step=epoch)
        mlflow.log_metric("val_f1", report["f1"], step=epoch)
        mlflow.log_metric("val_macro_f1", report["macro_f1"], step=epoch)
        mlflow.log_metric("val_weighted_f1", report["weighted_f1"], step=epoch)
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
        val_f1_arr[epoch] = report["macro_f1"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {report['loss']:.4f} | "
            f"Accuracy: {report['accuracy']:.4f} | "
            f"Precision: {report['precision']:.4f} | "
            f"Recall: {report['recall']:.4f} | "
            f"F1: {report['f1']:.4f} | "
            f"Macro F1: {report['macro_f1']:.4f} | "
            f"Weighted F1: {report['weighted_f1']:.4f} | "
            f"ROC-AUC: {report['roc_auc']:.4f}"
        )


    end = datetime.now()
    hours, remainder = divmod((end - start).total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")


    report, cm = evaluate_transformer_model(model, val_dataloader, device=device)
  
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Loss: {report['loss']:.4f}")
    print(f"ROC AUC: {report['roc_auc']:.4f}")
    print(report)



    model_name = config["model"]["name"]
    base_dir = config["output"]["base_dir"]

    run_dir = os.path.join(
        base_dir,
        model_name,
        run_id
    )

    save_outputs(run_dir, model, tokenizer, report, config, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr)


    #save_outputs_torch(run_dir, model, tokenizer, report, config, train_loss_arr, val_loss_arr, val_accuracy_arr, val_roc_auc_arr, val_f1_arr)
    #confusion_matrix_heatmap(run_dir)

    mlflow.log_artifacts(os.path.join(run_dir))

    mlflow.end_run()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    df = pd.read_parquet(config["df"])

    main(df, config)