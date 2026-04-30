from sklearn.metrics import classification_report, roc_auc_score
import torch
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def evaluate(model, data_loader):
    model.eval()
    model.to("cpu")

    predictions = []
    actual_labels = []
    positive_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to("cpu")
            attention_mask = batch["attention_mask"].to("cpu")
            labels = batch["label"].to("cpu")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            positive_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(actual_labels, predictions)

    report = classification_report(
        actual_labels,
        predictions,
        output_dict=True
    )

    auc = roc_auc_score(actual_labels, positive_probs)

    report["accuracy"] = accuracy
    report["roc_auc"] = float(auc)

    return accuracy, report

def evaluate_logreg(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    report["accuracy"] = accuracy
    report["roc_auc"] = float(auc)

    return accuracy, report