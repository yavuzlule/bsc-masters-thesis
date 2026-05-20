import pandas as pd
import ast
from datasets import Dataset
import torch
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import TrainingArguments
from transformers import Trainer


def parse_list_column(x):
    """
    Converts stringified list → list of strings.
    Handles:
    - valid Python list strings
    - already-list inputs
    - NaN / malformed cases
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except Exception:
            # fallback: treat as single text blob
            return [x]

    return []

def join_list_text(lst):
    """
    Joins list of strings into a single text field.
    """
    lst = [str(i).strip() for i in lst if str(i).strip()]
    return " ".join(lst)


def align_labels(text, start, end, tokenizer, max_length=512):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    offsets = encoding["offset_mapping"]
    labels = []

    for (s, e) in offsets:
        # special tokens
        if s == 0 and e == 0:
            labels.append(-100)
            continue

        # token overlaps title span
        if s >= start and e <= end:
            labels.append(1)
        else:
            labels.append(0)

    encoding["labels"] = labels
    return encoding



def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)

    true_preds = []
    true_labels = []

    for pred, label in zip(preds, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_preds.append(p_i)
                true_labels.append(l_i)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="binary"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }




def infer(text, title, start, tokenizer):

    
    end = len(title)


    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    offsets = encoding["offset_mapping"][0]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]


    labels = []

    for s, e in offsets:
        if s == 0 and e == 0:
            labels.append(-100)
        elif s >= start and e <= end:
            labels.append(1)
        else:
            labels.append(0)

    labels = torch.tensor([labels])

    print(text, title, labels)


if __name__ == "__main__":

    df = pd.read_csv("/Users/yavuzlule/Desktop/bsc-relish/data/external/recipe1m/full_dataset.csv")

    df = df[:100]
    df["ingredients"] = df["ingredients"].apply(parse_list_column).apply(join_list_text)
    df["directions"]  = df["directions"].apply(parse_list_column).apply(join_list_text)


    df = df[["title", "ingredients", "directions"]].copy()

    df["text"] = df["title"].astype(str) + " " + df["ingredients"].astype(str) + " " + df["directions"].astype(str)
    df["title"] = df["title"].astype(str)

    df["title_start"] = 0
    df["title_end"] = df["title"].str.len()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    processed = [
        align_labels(
            row["text"],
            row["title_start"],
            row["title_end"],
            tokenizer
        )
        for _, row in df.iterrows()
    ]



    dataset = Dataset.from_list(processed)


    id2label = {0: "O", 1: "TITLE"}
    label2id = {"O": 0, "TITLE": 1}

    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]



    data_collator = DataCollatorForTokenClassification(tokenizer)


    args = TrainingArguments(
        output_dir="./xlmr-title-ner",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_steps=50,
        metric_for_best_model="f1"
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    run_dir = "/Users/yavuzlule/Desktop/bsc-relish/results/xlmroberta-title"
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    trainer.train()

    text = "Chocolate Cake In a bowl mix flour sugar eggs. Bake for 30 minutes."
    title = "Chocolate Cake"

    start = 0
    infer(text, title, start)