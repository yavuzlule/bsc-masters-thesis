import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from pathlib import Path

def visualize_report(folder_path):
    """
    Visualize the classification report as a heatmap.

    Args:
        report (str): The classification report string from sklearn.

    Returns:
        None: Displays a heatmap of the classification report.

    Example report format:
    {
    "0": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 807.0
    },
    "1": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 24.0
    },
    "accuracy": 1.0,
    "macro avg": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 831.0
    },
    "weighted avg": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 831.0
    },
    "roc_auc": 1.0
    }
    """
    # Parse the report into a DataFrame

    print("Loading classification report...")

    report_path = os.path.join(folder_path, "metrics.json")
    with open(report_path, 'r') as f:
        report = f.read()
    lines = json.loads(report)
    data = []
    for key, metrics in lines.items():
        if key not in ["accuracy", "macro avg", "weighted avg", "roc_auc"]:
            data.append({
                'class': key,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })

    df_report = pd.DataFrame(data)

    # Set class as index for better visualization
    df_report.set_index('class', inplace=True)

    # Create a heatmap of precision, recall, and f1-score
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues', fmt=".4f", vmin=0, vmax=1)
    title = "/".join(report_path.split("/")[-3:-1])  # Extract filename from path
    title = f"Classification Report ({title})"
    title = title.replace("_", " ").replace(".json", "")
    plt.title(title)
    plt.savefig(os.path.join(folder_path, "classification_report.png"))
    print("Classification report heatmap saved to:", os.path.join(folder_path, "classification_report.png"))

def confusion_matrix_heatmap(folder_path):
    """
    Visualize the confusion matrix as a heatmap.

    Args:
        cm (array-like): Confusion matrix values.
        classes (list): List of class names corresponding to the confusion matrix.
        folder_path (str): Path to save the heatmap image.
    Returns:
        None: Displays a heatmap of the confusion matrix.
    """

    print("Creating confusion matrix heatmap...")

    report_path = os.path.join(folder_path, "metrics.json")
    with open(report_path, 'r') as f:
        report = f.read()
    lines = json.loads(report)
    cm = [[lines['0']['support'], 0], [0, lines['1']['support']]]
    classes = ['Non-recipe', 'Recipe']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Reds', xticklabels=classes, yticklabels=classes)

    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = "/".join(report_path.split("/")[-3:-1])  # Extract filename from path
    title = f"Confusion Matrix ({title})"
    title = title.replace("_", " ").replace(".json", "")
    plt.title(title)
    plt.savefig(os.path.join(folder_path, "confusion_matrix.png"))

    print("Confusion matrix heatmap saved to:", os.path.join(folder_path, "confusion_matrix.png"))


def main(folder_path):
    # Example usage

    visualize_report(folder_path)
    confusion_matrix_heatmap(folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to run directory")

    args = parser.parse_args()

    main(args.run_dir)