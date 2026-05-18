import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from pathlib import Path

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(folder_path):
    metrics_path = os.path.join(folder_path, "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)


# -------------------------------
# Classification Report Heatmap
# -------------------------------
def visualize_report(folder_path):
    print("Loading classification report...")

    report = load_metrics(folder_path)

    # Extract only per-class metrics
    rows = []
    for key, val in report.items():
        if isinstance(val, dict) and "precision" in val:
            rows.append({
                "class": key,
                "precision": val["precision"],
                "recall": val["recall"],
                "f1-score": val["f1-score"],
                "support": val["support"]
            })

    df = pd.DataFrame(rows).set_index("class")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[["precision", "recall", "f1-score"]],
        annot=True,
        fmt=".4f",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    title = f"Classification Report | Acc={report['validation_accuracy']:.4f} | AUC={report['roc_auc']:.4f}"
    plt.title(title)

    out_path = os.path.join(folder_path, "classification_report.png")
    plt.savefig(out_path)
    plt.close()

    print("Saved:", out_path)


# -------------------------------
# ❗ Proper Confusion Matrix Handling
# -------------------------------
def confusion_matrix_heatmap(folder_path):
    """
    This version EXPECTS a real confusion matrix saved separately.
    If you don't have one, DO NOT fabricate it from support.
    """

    cm_path = os.path.join(folder_path, "confusion_matrix.npy")

    if not os.path.exists(cm_path):
        print("⚠️ No confusion_matrix.npy found. Skipping confusion matrix.")
        return

    cm = np.load(cm_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    out_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()

    print("Saved:", out_path)


# -------------------------------
# Training Curves Visualization
# -------------------------------
def plot_training_curves(folder_path):
    print("Loading training curves...")

    npz_path = os.path.join(folder_path, "epochs.npz")

    if not os.path.exists(npz_path):
        print("⚠️ epochs.npz not found. Skipping curves.")
        return

    data = np.load(npz_path)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_acc = data["accuracy"]
    val_auc = data["roc_auc"]
    val_f1 = data["f1_score"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ---- Loss curve ----
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "loss_curve.png"))
    plt.close()

    # ---- Accuracy curve ----
    plt.figure()
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "accuracy_curve.png"))
    plt.close()

    # ---- ROC AUC ----
    plt.figure()
    plt.plot(epochs, val_auc, label="Validation ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("Validation ROC AUC")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "roc_auc_curve.png"))
    plt.close()

    # ---- F1 Score ----
    plt.figure()
    plt.plot(epochs, val_f1, label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "f1_curve.png"))
    plt.close()

    print("Training curves saved.")


# -------------------------------
# Main
# -------------------------------
def main(folder_path):
    visualize_report(folder_path)
    confusion_matrix_heatmap(folder_path)
    plot_training_curves(folder_path)


def main(folder_path):
    # Example usage
    visualize_report(folder_path)
    confusion_matrix_heatmap(folder_path)
    plot_training_curves(folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to run directory")

    args = parser.parse_args()

    main(args.run_dir)