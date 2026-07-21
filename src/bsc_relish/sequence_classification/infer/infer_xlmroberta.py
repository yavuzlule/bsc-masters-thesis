from datetime import datetime
from time import time
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from safetensors.torch import load_file
from pathlib import Path

def convert_tf_to_pytorch_layernorm(state_dict):
    """
    Convert TensorFlow LayerNorm parameters (gamma, beta) to PyTorch (weight, bias).
    
    Args:
        state_dict (dict): State dictionary with TensorFlow naming
    
    Returns:
        dict: Converted state dictionary with PyTorch naming
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Replace gamma with weight
        if 'LayerNorm.gamma' in key:
            new_key = key.replace('LayerNorm.gamma', 'LayerNorm.weight')
            new_state_dict[new_key] = value
        # Replace beta with bias
        elif 'LayerNorm.beta' in key:
            new_key = key.replace('LayerNorm.beta', 'LayerNorm.bias')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def load_roberta_model(model_path):
    """
    Load a RoBERTa-base model from safetensors format.
    Handles TensorFlow to PyTorch conversion.
    
    Args:
        model_path (str): Path to the safetensors model file
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model architecture...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        use_safetensors=True,
    )        
    print("Loading safetensors weights...")
    #state_dict = load_file(model_path)
    
    # Convert TensorFlow naming to PyTorch
    print("Converting TensorFlow parameters to PyTorch format...")
    #state_dict = convert_tf_to_pytorch_layernorm(state_dict)
    
    # Load with strict=False to handle any remaining mismatches
    #model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer

def get_device():
    """
    Determine the best available device (GPU or CPU).
    
    Returns:
        torch.device: The device to use for inference
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def predict_single_text(text, model, tokenizer, device, max_length=512):
    """
    Get prediction probability for a single text.
    
    Args:
        text (str): Text to classify
        model: The RoBERTa model
        tokenizer: The tokenizer
        device (torch.device): Device to use
        max_length (int): Maximum token length
    
    Returns:
        float: Probability of belonging to label 1
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # Return probability of label 1 (second class)
    label_1_probability = probabilities[0][1].item()
    
    return label_1_probability


def infer_roberta_optimized(
    df,
    model_path,
    column_name="chunk_text",
    batch_size=32,
    max_length=256,
):
    import torch
    import numpy as np
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

    df = df[df["label"].isin([0, 1])].copy()

    print("Loading model and tokenizer...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    texts = df[column_name].fillna("").astype(str).tolist()
    all_probabilities = []

    print(f"Processing {len(texts)} texts in batches of {batch_size}...")

    batch_starts = range(0, len(texts), batch_size)

    for batch_start in tqdm(
        batch_starts,
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Inference",
    ):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        try:
            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            label_1_probs = probs[:, 1].cpu().numpy()

            all_probabilities.extend(label_1_probs)

        except Exception as e:
            tqdm.write(f"Batch error {batch_start}-{batch_end}: {e}")
            all_probabilities.extend([np.nan] * len(batch_texts))

    df["xlmroberta-base-proba"] = all_probabilities

    return df


def debug_logits(df, model_path, column_name='chunk_text', num_samples=5):
    """Check raw logits being produced"""
    print("Loading model...")
    model, tokenizer = load_roberta_model(model_path)
    
    print(f"\nChecking logits for first {num_samples} samples:\n")

    for idx in range(min(num_samples, len(df))):
        text = df[column_name].iloc[idx]

        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        print(f"Sample {idx}:")
        print(f"  Text preview: {text[:80]}...")
        print(f"  Logits: [{logits[0].item():.4f}, {logits[1].item():.4f}]")
        print(f"  Probabilities: [{probs[0].item():.4f}, {probs[1].item():.4f}]")
        print(f"  Logit difference: {abs(logits[0].item() - logits[1].item()):.4f}")
        print()



# Usage
if __name__ == "__main__":
    # Load your DataFrame
    data_path = "/media/M2_disk/yavuz/bsc-masters-thesis/data/interim/chunked_256/2026-06-17_06-46-11-multilingual-xlm-256/dataset.parquet"
    df = pd.read_parquet(data_path)
    model_path='/media/M2_disk/yavuz/bsc-masters-thesis/results/FacebookAI/xlm-mlm-en-2048/2026-06-12_14-39-54'
    # Option 1: Single-by-single processing (slower, more memory efficient)
    # df = infer_roberta_batch(df, model_path='path/to/model.safetensors')
    #debug_logits(df, model_path, num_samples=5)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Option 2: Batch processing (faster, recommended)
    df = infer_roberta_optimized(df, model_path=model_path, batch_size=32)
    save_data_path = f"data/test/miriam-test-xlm-proba-{run_id}.parquet"
    print(f"Saving results to: {save_data_path}")
    # Save results
    df.to_parquet(save_data_path)