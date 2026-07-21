import asyncio
import pandas as pd
import fasttext
from bsc_relish.MoE.boundary_extractor import boundary_extractor
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bsc_relish.MoE.utils import filter_paragraphs, filter_similar_paragraphs, normalize_document, process_input, remove_duplicates
from bsc_relish.agentic.ingredient_extractor import ingredient_extractor
from bsc_relish.agentic.main_gepeto import save_report
from bsc_relish.agentic.metrics import evaluate_list
from bsc_relish.sequence_classification.train.train_bert import load_config
import os
import json
import asyncio
import pandas as pd
from tqdm import tqdm


def build_dataframe(paragraphs):
    """
    Build the working dataframe from a list of paragraphs. Every paragraph
    is treated as already-relevant (proba = 1) and used directly as the
    extended_chunk to run through the boundary extractor.
    """
    return pd.DataFrame({
        "proba": [1] * len(paragraphs),
        "extended_chunk": paragraphs,
    })


async def process_paragraphs(paragraphs):
    """
    Run the extraction pipeline on a list of paragraphs (from a single JSON
    file's "content" -> "paragraphs" entry) and return a list of extracted
    recipe strings.
    """
    output_df = build_dataframe(paragraphs)
    output_df["extracted_recipe"] = ""

    for index, row in tqdm(output_df.iterrows(), total=len(output_df), leave=False):
        extended_chunk = row["extended_chunk"]
        try:
            result = await boundary_extractor.run(extended_chunk)
            recipe = result.output
            output_df.at[index, "extracted_recipe"] = recipe
        except Exception as e:
            print(f"Error processing chunk at index {index}: {e}")

    output_df = remove_duplicates(output_df)

    recipes = output_df[
        output_df["extracted_recipe"].notnull()
        & (output_df["extracted_recipe"] != "")
    ]["extracted_recipe"].tolist()

    return recipes


def chunk_paragraphs(paragraphs, chunk_size=256):
    """
    Chunk paragraphs into smaller pieces of a specified size.
    """
    chunks = []
    for paragraph in paragraphs:
        words = paragraph.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


async def main(root_folder):
    all_recipes = []

    if not os.path.isdir(root_folder):
        raise NotADirectoryError(f"root_folder does not exist or is not a directory: {root_folder}")

    subfolders = sorted(
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    )

    for folder in tqdm(subfolders, desc="Processing folders"):
        json_files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))

        if not json_files:
            print(f"No JSON file found in {folder}, skipping.")
            continue
        if len(json_files) > 1:
            print(f"Multiple JSON files found in {folder}, using {json_files[0]}.")

        json_path = os.path.join(folder, json_files[0])

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            print(f"Skipping {json_path}: could not read/parse JSON ({e}).")
            continue

        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            print(f"Unexpected structure in {json_path}, skipping.")
            continue

        paragraphs = data[0].get("content", {}).get("paragraphs")

        if not paragraphs:
            print(f"No 'paragraphs' found in {json_path}, skipping.")
            continue

        chunks = chunk_paragraphs(paragraphs)

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)} in folder {folder}: {chunk[:100]}...")

            try:
                # process_paragraphs expects a list of paragraphs, not a raw string.
                recipes = await process_paragraphs([chunk])
            except Exception as e:
                print(f"Error processing chunk {i + 1} in folder {folder}: {e}")
                continue

            all_recipes.extend(recipes)

    print(f"Total recipes before filtering: {len(all_recipes)}")
    all_recipes = filter_paragraphs(all_recipes)
    print(f"Total recipes after filtering: {len(all_recipes)}")
    all_recipes = filter_similar_paragraphs(all_recipes)
    print(f"Total recipes after filtering similar paragraphs: {len(all_recipes)}")
    all_recipes = list(filter(lambda x: len(x.split()) > 1, all_recipes))
    print(f"Total recipes after filtering short recipes: {len(all_recipes)}")
    print(f"Total recipes extracted: {len(all_recipes)}")

    return all_recipes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    root_folder = config.get("input_data")
    if not root_folder:
        raise ValueError("config.yaml is missing required key 'input_data'")

    recipes = asyncio.run(main(root_folder))
    recipes_df = pd.DataFrame({"extracted_recipe": recipes})

    os.makedirs(root_folder, exist_ok=True)
    save_path = os.path.join(root_folder, "extracted_recipes.csv")
    recipes_df.to_csv(save_path, index=False)
    print(f"Saved {len(recipes_df)} recipes to {save_path}")