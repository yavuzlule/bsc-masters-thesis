import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def evaluate(model, data_loader, device="cpu"):
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
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            positive_probs.extend(probs[:, 1].cpu().numpy())

    # Final metrics
    validation_loss = total_loss / len(data_loader)
    validation_accuracy = accuracy_score(actual_labels, predictions)
    auc = roc_auc_score(actual_labels, positive_probs)

    cm = confusion_matrix(actual_labels, predictions)

    report = classification_report(
        actual_labels,
        predictions,
        output_dict=True
    )

    # Add summary fields
    report["accuracy"] = validation_accuracy
    report["roc_auc"] = float(auc)
    report["loss"] = validation_loss

    return report, cm



def evaluate_logreg(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    report["accuracy"] = accuracy
    report["roc_auc"] = float(auc)

    return accuracy, report