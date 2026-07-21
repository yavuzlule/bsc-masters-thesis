import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import yaml
import json
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from bsc_relish.preprocess.chunk.chunk import expand_chunks
import yaml
from joblib import dump
from bsc_relish.utils.utils import evaluate_classical_model, get_embedding


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



def save_outputs(model, run_dir, report, config):

    metrics_path = os.path.join(run_dir, "metrics.json")
    logs_path = os.path.join(run_dir, "logs.txt")
    config_path = os.path.join(run_dir, "config.yaml")
    model_path = os.path.join(run_dir, "model.pickle")

    dump(model, model_path)


    # Predictions
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with open(logs_path, "w") as f:
        f.write(f"Validation Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {report['roc_auc']:.4f}\n")
        f.write(json.dumps(report, indent=2))

    print("Saved to: " + run_dir)



# -------------------------
# Main
# -------------------------
import os
from datetime import datetime


def main(df):
    config = load_config("configs/xgboost.yaml")

    # Load data
    train_df = df
    target = "label"

    print(len(train_df))
    preprocess_config = load_config("/media/M2_disk/yavuz/bsc-masters-thesis/configs/preprocess.yaml")



    rows = train_df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")
    model_name = config["model"]["name"]


    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Obtain embeddings (allminilm-embeddings-v2)
    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        preprocess_config["preprocessing"]["chunking"]["max_words"],
    )
    train_df["embedding"] = train_df["text"].apply(lambda x: get_embedding(x, embedding_model))






    # Reset indices
    train_df = train_df.reset_index(drop=True)

    #X = train_df.drop(columns=[target])
    X = train_df["embedding"].tolist() 
    y = train_df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        stratify=y if config["training"]["stratify"] else None,
        random_state=42,
    )

    # Build pipeline
    #preprocessor = build_preprocessor(config)
    model = load_model(config["model"]["type"], config["model"]["params"])
    pipeline = Pipeline([
        ("model", model),
    ])


    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    report, cm = evaluate_classical_model(pipeline, X_test, y_test)

    # Save outputs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = config["model"]["name"]
    base_dir = config["output"]["base_dir"]

    run_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)


    save_outputs(model, run_dir, report, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    df = pd.read_parquet(config["df"])
    main(df)