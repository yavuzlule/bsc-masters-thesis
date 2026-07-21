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
"""

Pipeline For Mixture of Experts (MoE) Model:
1. Load the tokenizer and model using Hugging Face's transformers library.
2. Define a function to process input text through the model and return the output.
3. Detect the language of the input text and route it to the appropriate expert model based on the detected language.
4. Return the output from the selected expert model.
5. Output the final JSON result containing the model's predictions and any relevant metadata.

"""



from tqdm import tqdm

async def main(input_data_path):
    with open(input_data_path, 'r') as f:
        input_data = f.read()
    
    df = pd.read_parquet("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/agentic/data/recipe1m_10000.parquet")
    num_recipes = 100
    df = df[34:num_recipes]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row["directions"]
        reference_ingredients = row["NER"].split(",")  # Assuming the reference ingredients are stored as a comma-separated string
        print(text)
        ingredients = await ingredient_extractor.run(text)
        ingredients = ingredients.output.ingredients
        print("Ingredients:", ingredients)
        ingredients_report = evaluate_list(reference_ingredients, ingredients)

        base_dir = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/ner"
        ingredients_path = os.path.join(base_dir, f"ingredient_extraction_llm_{num_recipes}.json")
        save_report(ingredients_report, ingredients_path)
    
    
    """
    for index, row in tqdm(output_df.iterrows(), total=len(output_df)):
        if row['proba'] < 0.5:
            output_df.at[index, "extracted_recipe"] = ""
            continue

        extended_chunk = row["extended_chunk"]
        result = await boundary_extractor.run(extended_chunk)
        recipe = result.output
        output_df.at[index, "extracted_recipe"] = recipe

    output_df = remove_duplicates(output_df)

    recipes = output_df[
        output_df["extracted_recipe"].notnull() &
        (output_df["extracted_recipe"] != "")
    ]["extracted_recipe"].tolist()

    print(f"Total recipes before filtering: {len(recipes)}")
    recipes = filter_paragraphs(recipes)
    print(f"Total recipes after filtering: {len(recipes)}")
    recipes = filter_similar_paragraphs(recipes)
    print(f"Total recipes after filtering similar paragraphs: {len(recipes)}")
    recipes = list(filter(lambda x: len(x.split()) > 1, recipes))  # Filter out recipes with less than 2 words
    print(f"Total recipes after filtering short recipes: {len(recipes)}")
    print(f"Total recipes extracted: {len(recipes)}")
    """  
    return recipes



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    input_data_folder = config["input_data"]
    
    txt_files = [f for f in os.listdir(input_data_folder) if f.endswith('.txt')]
    input_data_path = os.path.join(input_data_folder, txt_files[0])  # Assuming there's at least one .txt file in the folder
    recipes = asyncio.run(main(input_data_path))
    recipes_df = pd.DataFrame({"extracted_recipe": recipes})

    os.makedirs(input_data_folder, exist_ok=True)
    save_path = os.path.join(input_data_folder, "extracted_recipes.csv")
    recipes_df.to_csv(save_path, index=False)