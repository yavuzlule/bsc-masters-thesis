import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    print("Loading model architecture...")
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    print("Loading safetensors weights...")
    state_dict = load_file(model_path)
    
    # Convert TensorFlow naming to PyTorch
    print("Converting TensorFlow parameters to PyTorch format...")
    state_dict = convert_tf_to_pytorch_layernorm(state_dict)
    
    # Load with strict=False to handle any remaining mismatches
    model.load_state_dict(state_dict, strict=False)
    
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


def infer_bert_batch(df, model_path, column_name='chunk_text', batch_size=8, max_length=512):
    """
    Infer BERT probabilities for all texts in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        model_path (str): Path to the safetensors model file
        column_name (str): Name of the column containing text to classify
        batch_size (int): Batch size for processing
        max_length (int): Maximum token length
    
    Returns:
        pd.DataFrame: DataFrame with added 'Roberta-base-proba' column
    """
    print("Loading model and tokenizer...")
    model, tokenizer = load_roberta_model(model_path)
    device = get_device()
    model.to(device)
    
    print(f"Processing {len(df)} texts...")
    probabilities = []
    
    for idx, text in enumerate(df[column_name]):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)}")
        
        try:
            prob = predict_single_text(text, model, tokenizer, device, max_length)
            probabilities.append(prob)
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            probabilities.append(None)
    
    # Add column to DataFrame
    df['Roberta-base-proba'] = probabilities
    
    print("Done!")
    return df


def infer_bert_optimized(df, model_path, column_name='chunk_text', batch_size=32, max_length=512):
    """
    Optimized batch inference using vectorized operations.
    More efficient than processing one-by-one.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        model_path (str): Path to the safetensors model file
        column_name (str): Name of the column containing text to classify
        batch_size (int): Batch size for processing
        max_length (int): Maximum token length
    
    Returns:
        pd.DataFrame: DataFrame with added 'Roberta-base-proba' column
    """
    print("Loading model and tokenizer...")
    model, tokenizer = load_roberta_model(model_path)
    device = get_device()
    model.to(device)
    
    texts = df[column_name].tolist()
    all_probabilities = []
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        print(f"  Batch {batch_start}-{batch_end}")
        
        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
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
            
            # Get probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Extract label 1 probabilities
            label_1_probs = probabilities[:, 1].cpu().numpy().tolist()
            all_probabilities.extend(label_1_probs)
        
        except Exception as e:
            print(f"  Error processing batch {batch_start}-{batch_end}: {e}")
            all_probabilities.extend([None] * len(batch_texts))
    
    # Add column to DataFrame
    df['bert-base-uncased-proba'] = all_probabilities
    
    print("Done!")
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
        print(outputs.logits.shape)
        print(outputs.logits[:5])
        print(probs[:5])
        print()



# Usage
if __name__ == "__main__":
    # Load your DataFrame
    df = pd.read_parquet('/Users/yavuzlule/Desktop/bsc-relish/data/interim/b2drop_dataset.parquet')
    model_path='/Users/yavuzlule/Desktop/bsc-relish/results/bert-base-uncased/2026-04-30_14-20-28/model.safetensors'
    # Option 1: Single-by-single processing (slower, more memory efficient)
    #df = infer_bert_batch(df, model_path='path/to/model.safetensors')
    #debug_logits(df, model_path, num_samples=5)
    
    # Option 2: Batch processing (faster, recommended)
    df = infer_bert_optimized(df, model_path=model_path, batch_size=32)

    # Save results
    df.to_parquet('/Users/yavuzlule/Desktop/bsc-relish/data/interim/b2drop_v1/output_with_probabilities_bert.parquet')