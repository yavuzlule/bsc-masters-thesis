import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import torch


def evaluate(model, data_loader, device):
    model.eval()
    model.to(device)

    predictions = []
    actual_labels = []
    positive_probs = []

    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=batch["lang_ids"].to(device)
            )

            logits = outputs

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

            # Binary classification ROC-AUC
            positive_probs.extend(probs[:, 1].cpu().numpy())

    # Loss
    validation_loss = total_loss / len(data_loader)

    # Metrics
    validation_accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, zero_division=0)
    recall = recall_score(actual_labels, predictions, zero_division=0)
    f1 = f1_score(actual_labels, predictions, zero_division=0)

    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(actual_labels, positive_probs)

    cm = confusion_matrix(actual_labels, predictions)

    report = classification_report(
        actual_labels,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)
    report["loss"] = float(validation_loss)

    return report, cm


def evaluate_logreg(model, X_test, y_test):
    total_loss = 0.0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    

    # Metrics
    validation_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    macro_f1 = f1_score(
        y_test, 
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, 
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)

    return report, cm

def evaluate_svm(model, X_test, y_test):
    total_loss = 0.0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    

    # Metrics
    validation_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    macro_f1 = f1_score(
        y_test, 
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, 
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)

    return report, cm


def evaluate_xgboost(model, X_test, y_test):
    total_loss = 0.0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    

    # Metrics
    validation_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    macro_f1 = f1_score(
        y_test, 
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, 
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Add top-level summary metrics
    report["accuracy"] = float(validation_accuracy)
    report["precision"] = float(precision)
    report["recall"] = float(recall)
    report["f1"] = float(f1)
    report["macro_f1"] = float(macro_f1)
    report["weighted_f1"] = float(weighted_f1)
    report["roc_auc"] = float(auc)

    return report, cm