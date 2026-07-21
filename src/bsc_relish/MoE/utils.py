

"""

1. PREPROCESSING
    - Normalize Document: Convert the input (image, pdf, etc.) to string format.
   - Chunking: Split the input text into manageable chunks if it exceeds the model's maximum input length.
   - Tokenization: Split the input text into tokens using the appropriate tokenizer.
   - Language Detection: Identify the language of the input text.
   - Normalize the text (e.g., lowercasing, removing punctuation) if necessary.

2. EXPERT SELECTION
   - Route the tokenized input to the appropriate expert model based on the detected language.

3. INFERENCE
   - Pass the input through the selected expert model and obtain the output.

4. POSTPROCESSING
   - Convert the model output into a human-readable format.
   - Include any relevant metadata (e.g., language, model used) in the final output.

   
   PIPELINE FOR Mixture of Experts (MoE) MODEL:
   INPUT (text, image, pdf) -> TEXT -> CHUNK -> LANGUAGE DETECTION -> TOKENIZATION -> EXPERT SELECTION -> INFERENCE -> OUTPUT (JSON)

"""


import os
import shutil
from huggingface_hub import hf_hub_download
import pandas as pd
import fasttext
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bsc_relish.sequence_classification.train.train_bert import load_config
import torch

LANGUAGE_ORDER = ['english', 
                  'old german', 
                  'catalan', 
                  'multiple', 
                  'italian', 
                  'latin',
                  'french', 
                  'old french', 
                  'venetian/italian', 
                  'old danish', 
                  'middle dutch', 
                  'middle low german', 
                  'early modern english', 
                  'middle french', 
                  'spanish', 
                  'german', 
                  'dutch']


LANGUAGE_CODES_MAP = {
    'english': 'eng',
    'old german': 'de',
    'catalan': 'ca',
    'multiple': 'mul',
    'italian': 'it',
    'latin': 'la',
    'french': 'fr',
    'old french': 'fro',
    'venetian/italian': 'it',
    'old danish': 'da',
    'middle dutch': 'nl',
    'middle low german': 'nds',
    'early modern english': 'en',
    'middle french': 'fr',
    'spanish': 'es',
    'german': 'de',
    'dutch': 'nl'
}

cached_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
detect_language_model = fasttext.load_model(cached_path)
save_dir = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/MoE/models"
os.makedirs(save_dir, exist_ok=True)



final_path = os.path.join(save_dir, "model.bin")
shutil.copy(cached_path, final_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_document(input_data):
    # Convert the input (image, pdf, etc.) to string format.
    # This is a placeholder for actual implementation.
    # For example, if input_data is an image, use OCR to extract text.
    # If input_data is a PDF, use a PDF parser to extract text.
    return str(input_data)

def detect_language(text):
    # Use the fastText model to detect the language of the input text
    predictions = detect_language_model.predict(text)
    language_code = predictions[0][0].replace("__label__", "")
    language_code = language_code.replace("_Latn", "")
    return language_code

def pick_expert_model(language, results):
    best_f1_score = 0
    best_model = None
    for model_name, metrics in results.items():
        f1_score = metrics.get("f1", 0)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model_name
    return best_model

def infer_row(row, tokenizer, model):
    inputs = tokenizer(row["text"], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    row["model_output"] = outputs.logits.detach().numpy().tolist()
    return row

def remove_duplicates(df):
    # Remove duplicate rows based on the 'extracted_recipe' column
    df = df.drop_duplicates(subset=['extracted_recipe'])
    return df


def filter_paragraphs(paragraphs):
    # convert to list of sets + keep original index
    parsed = [(i, set(p.split()), p) for i, p in enumerate(paragraphs)]

    # sort by size descending (keep longest first)
    parsed.sort(key=lambda x: len(x[1]), reverse=True)

    kept = []
    kept_sets = []

    for i, s, original in parsed:
        # check if already contained in a kept paragraph
        if not any(s.issubset(k) for k in kept_sets):
            kept.append(original)
            kept_sets.append(s)

    return kept

def jaccard_similarity(a, b):
    return len(a & b) / len(a | b)


def filter_similar_paragraphs(paragraphs, threshold=0.9):
    # convert paragraphs to word sets
    parsed = [(p, set(p.split())) for p in paragraphs]

    # sort by length (keep longest first)
    parsed.sort(key=lambda x: len(x[1]), reverse=True)

    kept = []
    kept_sets = []

    for text, s in parsed:
        is_similar = False

        for ks in kept_sets:
            if jaccard_similarity(s, ks) >= threshold:
                is_similar = True
                break

        if not is_similar:
            kept.append(text)
            kept_sets.append(s)

    return kept


def process_input(text):
    # Preprocessing
    # 1. Normalize Document: Convert the input (image, pdf, etc.) to string format.
    # 2. Chunking: Split the input text into manageable chunks if it exceeds the model's maximum input length.
    # For each chunk, perform the following steps:
        # 3. Language Detection: Identify the language of the input text.
        # 4. Tokenization: Split the input text into tokens using the appropriate tokenizer.
        # 5. Normalize the text (e.g., lowercasing, removing punctuation) if necessary.

    text = normalize_document(text)
    preprocess_config = load_config("/Users/yavuzlule/Desktop/bsc-relish/configs/preprocess.yaml")

    text = text.strip()
    text = text.replace("\n", " ")
    language_code = detect_language(text)
    language = next((lang for lang, code in LANGUAGE_CODES_MAP.items() if code == language_code), "unknown")

    df = pd.DataFrame({"text": [text], "language": [language]})
    """
    
    best_model_name = pick_expert_model(language, results={model_name: {"f1": 0.9}})  # Placeholder for actual results
    tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_name)
    """
    model_dir = "/Users/yavuzlule/Desktop/bsc-relish/results/distilbert/2026-04-30_07-30-33"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # Placeholder for actual model
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # Placeholder for actual tokenizer

    UNK_ID = len(LANGUAGE_ORDER)  # Assign a unique ID for unknown languages

    df["lang_id"] = df["language"].apply(
        lambda x: LANGUAGE_ORDER.index(x) if x in LANGUAGE_ORDER else UNK_ID
    )


    rows = df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")
    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        256,
        128
    )
        
    for row in expanded:
        inputs = tokenizer(row["chunk_text"], return_tensors="pt", padding=True)

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)

        label_1_probs = probs[:, 1].detach().cpu().numpy().tolist()[0]

        row["proba"] = label_1_probs

    for i in range(len(expanded)):
        if expanded[i]["proba"] < 0.5:
            continue
        extended_chunk = ""
        if i == 0:
            extended_chunk = expanded[i]["chunk_text"]
        elif i == len(expanded) - 1:
            extended_chunk = expanded[i - 1]["chunk_text"] + " " + expanded[i]["chunk_text"]
        else:
            extended_chunk = expanded[i - 1]["chunk_text"] + " " + expanded[i]["chunk_text"] + " " + expanded[i + 1]["chunk_text"]
        expanded[i]["extended_chunk"] = extended_chunk


    
    expanded = pd.DataFrame(expanded)
    return expanded

