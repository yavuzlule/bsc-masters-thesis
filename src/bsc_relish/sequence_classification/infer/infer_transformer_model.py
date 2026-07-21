from datetime import datetime
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from safetensors.torch import load_file
from pathlib import Path

from bsc_relish.preprocess.chunk.chunk import expand_chunks
from bsc_relish.utils.utils import load_config
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
    tokenizer = BertTokenizer.from_pretrained(model_path)

    print("Loading model architecture...")
    model = BertForSequenceClassification.from_pretrained(
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



def infer_bert_optimized(
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

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        use_safetensors=True,
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)

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

    df["bert-base-proba"] = all_probabilities

    return df


# Usage
if __name__ == "__main__":
    # Load your DataFrame
    config = load_config("/media/M2_disk/yavuz/bsc-masters-thesis/configs/distilbert.yaml")

    data_path = config["df"]
    df = pd.read_parquet(data_path)
    preprocess_config = load_config("/media/M2_disk/yavuz/bsc-masters-thesis/configs/preprocess.yaml")

    print(len(df))



    rows = df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")
    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        preprocess_config["preprocessing"]["chunking"]["max_words"],
    )

    df = pd.DataFrame(expanded)


    # Reset indices
    df = df.reset_index(drop=True)
    model_path='/media/M2_disk/yavuz/bsc-masters-thesis/results/distilbert-base-uncased/2026-07-03_08-21-11'
    # Option 1: Single-by-single processing (slower, more memory efficient)
    #df = infer_bert_batch(df, model_path='path/to/model.safetensors')
    #debug_logits(df, model_path, num_samples=5)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Option 2: Batch processing (faster, recommended)
    df = infer_bert_optimized(df, model_path=model_path, batch_size=32)
    save_data_path = f"data/test/multilingual-test-distilbert-proba-{run_id}.parquet"
    print(f"Saving results to: {save_data_path}")
    # Save results
    df.to_parquet(save_data_path)