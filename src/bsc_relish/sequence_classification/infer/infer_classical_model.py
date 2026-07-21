import os
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from bsc_relish.preprocess.chunk.chunk import expand_chunks
from joblib import load
from datetime import datetime
import os
from bsc_relish.utils.utils import evaluate_classical_model, get_embedding, load_config, save_outputs

def main(df):
    config = load_config("configs/logreg.yaml")

    target = config["data"]["target_column"]

    preprocess_config = load_config("/media/M2_disk/yavuz/bsc-masters-thesis/configs/preprocess.yaml")

    print(len(df))

    rows = df.rename(columns={"chunk_text": "text"}).to_dict(orient="records")
    model_name = config["model"]["name"]

    embedding_model_name = "allminilm-embeddings-v2"

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Obtain embeddings (allminilm-embeddings-v2)
    expanded = expand_chunks(
        preprocess_config,
        tokenizer,
        rows,
        preprocess_config["preprocessing"]["chunking"]["max_words"],
    )

    df["embedding"] = df["text"].apply(lambda x: get_embedding(x, embedding_model))


    


    # Reset indices
    df = df.reset_index(drop=True)
    # Features and labels
    X = df["embedding"].tolist()
    y = df[target]

    # Load trained model
    model = load(
        "/media/M2_disk/yavuz/bsc-masters-thesis/results/xgboost/2026-07-03_02-41-17/model.pickle"
    )
    df["xgboost-proba"] = model.predict_proba(X)[:, 1]


    # Evaluate
    report, cm = evaluate_classical_model(model, X, y)

    # Save outputs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = config["model"]["name"]
    base_dir = config["output"]["base_dir"]

    run_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_data_path = f"{run_dir}/multilingual-test-xgboost-proba-{run_id}.parquet"
    save_outputs(model, run_dir, report, config)
    df.to_parquet(save_data_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config("configs/xgboost.yaml")

    df = pd.read_parquet(config["df"])
    main(df)