import os
import numpy as np
import yaml
import json
import joblib
import importlib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import logging

import yaml



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
# Preprocessing
# -------------------------

def build_preprocessor(config: dict):
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    num_strategy = config["preprocessing"]["missing_values"]["numeric_strategy"]
    cat_strategy = config["preprocessing"]["missing_values"]["categorical_strategy"]

    scaler_method = config["preprocessing"]["scaling"]["method"]
    encoding_method = config["preprocessing"]["encoding"]["categorical"]

    # Numeric pipeline
    num_steps = [
        ("imputer", SimpleImputer(strategy=num_strategy)),
    ]

    if scaler_method == "standard":
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(num_steps)

    # Categorical pipeline
    cat_steps = [
        ("imputer", SimpleImputer(strategy=cat_strategy)),
    ]

    if encoding_method == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))

    categorical_pipeline = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# -------------------------
# Evaluation
# -------------------------

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# -------------------------
# Main
# -------------------------
import os
from datetime import datetime


def main(df):
    config = load_config("configs/logreg.yaml")

    # Load data
    train_df = df
    target = config["data"]["target_column"]

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
    print("\n\n\nMODEL LOADED\n\n")
    pipeline = Pipeline([
        ("model", model),
    ])
    print("\n\n\nPIPELINE\n\n")
    # Cross-validation
    if config["training"]["cross_validation"]["enabled"]:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=config["training"]["cross_validation"]["folds"],
            scoring=config["training"]["cross_validation"]["scoring"],
        )
        print(f"CV mean score: {scores.mean():.4f}")

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate(pipeline, X_test, y_test)
    print("Metrics:", metrics)

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
        joblib.dump(pipeline, model_path)


    # Predictions
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

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
    config = load_config("configs/logreg.yaml")

    df = pd.read_parquet(config["df"])
    main(df)