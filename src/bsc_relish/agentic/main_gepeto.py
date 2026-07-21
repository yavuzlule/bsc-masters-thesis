from bsc_relish.agentic.title_extractor import title_extractor
from bsc_relish.agentic.chunker import chunker
from bsc_relish.agentic.document_normalizer import document_normalizer
from bsc_relish.agentic.document_format_validator import document_validator
from bsc_relish.agentic.ingredient_extractor import ingredient_extractor
from bsc_relish.agentic.boundary_extractor import boundary_extractor
from bsc_relish.agentic.ingredient_extractor import ingredient_extractor
from bsc_relish.agentic.action_extractor import action_extractor
from datetime import datetime

from bsc_relish.agentic.metrics import evaluate_extraction
from bsc_relish.agentic.metrics import evaluate_list
import os
from pathlib import Path
import pypdf
import pandas as pd
from pydantic_ai.exceptions import UnexpectedModelBehavior
from bsc_relish.agentic.models import BoundaryInput
from time import time
import json
import os
from bsc_relish.gepeto.utils.gepeto.requests_ import send_prompt
import requests


DEFAULT_NORMALIZATION_VALUE = ""
DEFAULT_START_VALUE = 0
DEFAULT_END_VALUE = 0
DEFAULT_INGREDIENTS_VALUE = []
DEFAULT_ACTIONS_VALUE = []
DEFAULT_TITLE_VALUE = ""
BOUNDARY_EXTRACTION_PROMPT = """
    You extract recipe or instructions for assembling components from raw cooking text.

    You must return ONLY one valid output matching the schema.

    RULES:

    1. Identify the first recipe in the text.
    A recipe is defined as a contiguous block of sentences that includes actions that relate to preparing food.

    2. Once the first recipe starts, include all consecutive sentences that belong to it.

    3. Stop extraction immediately when:
    - a new recipe begins, OR
    - the text is no longer describing the same dish.

    4. If no complete recipe is found, return an empty string.

    5. Do NOT include explanations, labels, or commentary.

    OUTPUT RULE:
    - Return ONLY the extracted recipe text as a single string.
    - If nothing is found, return empty string.

    EXAMPLE 1:
        Input:
        Brown ground beef in frying pan. Chop onion and add to hamburger, cook until tender. Add all ingredients to crock-pot. 
        Cook on high for 4 to 6 hours. All ingredients can be altered to 
        personal taste. Zero-shot prompting means that the prompt used to interact with the model won't contain examples or demonstrations. 
        The zero-shot prompt directly instructs the model to perform a task without any additional examples to steer it.Line baking dish with beef. Wrap each breast with slice of bacon and place on top of beef. 
        Mix soup and sour cream and pour over chicken. Bake uncovered at 225° for 3 hours.

        Output:
        Brown ground beef in frying pan. Chop onion and add to hamburger, cook until tender. Add all ingredients to crock-pot. 
        Cook on high for 4 to 6 hours. All ingredients can be altered to 
        personal taste. 

    EXAMPLE 2:
        Input:
        Mix together 1 cup flour, margarine and sugar well. Put into 8-inch square pan and bake 15 minutes at 350°. Mix eggs, lemon juice, confectioners sugar, 1 tablespoon flour and soda well and spread over crust. 
        Bake 25 minutes more. Cool. Cut into squares and sprinkle with confectioners sugar. Yields 16 (2-inch) squares.
        Am I the only one that just gets annoyed by people saying "BG" for bad game at the end? Maybe it's just because 
        I come from the Starcraft community, but it was my understanding that GG didn't actually mean good game, it was just more of a polite way of ending the game, even if you didn't think it was a good game. 
        Honestly, if you feel so upset about the game going poorly, just don't say anything at all.
        Preheat oven to 400°. In a medium bowl, combine Fiesta Herb with Red Pepper soup mix and tortilla chips. 
        In a large plastic bag or bowl, combine chicken and egg, beaten with water, coating well. Dip chicken in 
        tortilla mixture, coating well. In a 15 1/2 x 10 1/2 x 1-inch jelly roll pan sprayed with no-stick cooking 
        spray, arrange chicken and drizzle with margarine. Bake, uncovered, for 12 minutes or until chicken is done. Makes about 24 chicken strips. Can be served with salsa.


        Output:
        Mix together 1 cup flour, margarine and sugar well. Put into 8-inch square pan and bake 15 minutes at 350°. Mix eggs, lemon juice, confectioners sugar, 1 tablespoon flour and soda well and spread over crust. 
        Bake 25 minutes more. Cool. Cut into squares and sprinkle with confectioners sugar. Yields 16 (2-inch) squares.
        """


class Recipe:
    def __init__(self, title: str, text: str, ingredients: list[str], actions: list[str]):
        self.title = title
        self.text = text
        self.ingredients = ingredients
        self.actions = actions


def save_report(report, file_path="reports.json"):
    """
    Appends a single evaluation report to a JSON file containing a list of reports.
    Creates the file if it does not exist.
    """

    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new report
    data.append(report)

    # Save back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_text(source: str | Path) -> str:
    # raw string
    if isinstance(source, str) and not Path(source).exists():
        return source

    path = Path(source)

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(f"Unsupported file type: {path}")

def split_into_chunks(text: str, chunk_size: int = 64) -> list[str]:
    words = text.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def check_integer(start):
    if not isinstance(start, int):
        raise TypeError("start must be int")
    return start

def concat_neighbors(texts, i):
    n = len(texts)
    parts = []

    for j in (i - 1, i, i + 1):
        if 0 <= j < n:
            parts.append(texts[j])

    return " ".join(parts)


def concat_first_n_txt(folder_path, n):
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.endswith(".txt")
    )[:n]



    parts = []
    for fname in files:
        print(fname)
        path = os.path.join(folder_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f.read())

    return "".join(parts)


def sliding_window_chunks(
    text: str,
    chunk_size: int = 64,
    window: int = 3,
) -> list[str]:
    """
    Split text into chunks, then return overlapping windows of chunks.

    window=3 means each output contains:
      previous chunk + current chunk + next chunk
    (when available).
    """
    words = text.split()

    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    n = len(chunks)
    half = window // 2

    return [
        " ".join(
            chunks[max(0, i - half): min(n, i + half + 1)]
        )
        for i in range(n)
    ]
async def main(df: pd.DataFrame) -> pd.DataFrame:
    
    base_dir = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/agentic/results"

    concatenated_directions = " ".join(
        df["title"].astype(str) + " " + df["directions"].astype(str)
    )
    df_titles = df["title"].astype(str).tolist()
    df_ingredients = df["ingredients"].astype(str).tolist()
    df_extracted = pd.DataFrame(columns=["title", "text", "ingredients", "actions"])
    chunks_list = []
    recipe_list = []
    # ==============================================
    # 1. STRING EXTRACTION
    # ==============================================
    raw_text = extract_text(concatenated_directions)

    print("Raw text:")
    print(raw_text)
    print("\n")
    
    chunks = sliding_window_chunks(raw_text, chunk_size=64, window=5)
    print("Chunks:")
    print(chunks)
    print("\n")
    for i in range(len(chunks)):
        chunk = chunks[i]    
        print("\n")
        print("="*100)
        print(f"\nCHUNK #{i+1}: \n{chunk}\n")
        print("="*100)
        print("\n")

        # ==============================================
        # 2. NORMALIZATION
        # ==============================================
        start_normalization = datetime.now()
        try:
            normalized = await document_normalizer.run(chunk)
            normalized_text = normalized.output.content
        except UnexpectedModelBehavior:
            normalized_text = DEFAULT_NORMALIZATION_VALUE
        chunks_list.append(normalized_text)
        end_normalization = datetime.now()
        normalization_time = end_normalization - start_normalization

        print("Normalized:")
        print(normalized_text)
        print("\n")
        

        normalization_report = evaluate_extraction(chunk, normalized_text)

        print("Normalization Report:")
        print(normalization_report)
        normalization_path = os.path.join(base_dir, f"normalization_results_{model_name}.json")
        save_report(normalization_report, normalization_path)

        # ==============================================
        # 3. VALIDATON
        # ==============================================

        validation_start = datetime.now()
        validation = await document_validator.run(normalized_text)
        if not validation.output:
            raise ValueError(validation.output['errors'])
        validation_end = datetime.now()
        validation_time = validation_end - validation_start
        print("Validation: ")
        print(validation)


        # ==============================================
        # 4. BOUNDARY EXTRACTION
        # ==============================================

        extended_chunk = normalized_text  # Use normalized text for boundary extraction

        print("Boundary Extraction:")
        print(extended_chunk)
        boundary_start = datetime.now()
        result = send_prompt(
            BOUNDARY_EXTRACTION_PROMPT.format(extended_chunk=extended_chunk),
            model="nemotron-3-super:120b",
        )
        recipe = result
        boundary_end = datetime.now()
        boundary_time = boundary_end - boundary_start

        boundary_extraction_report = evaluate_extraction(extended_chunk, recipe)

        print("Boundary Extraction Report:")
        print(boundary_extraction_report)
        boundary_extraction_path = os.path.join(base_dir, f"boundary_extraction_results_{model_name}.json")
        save_report(boundary_extraction_report, boundary_extraction_path)

        # ==============================================
        # 5. INGREDIENT EXTRACTION
        # ==============================================

        ingredient_start = datetime.now()
        try:
            ingredients = await ingredient_extractor.run(recipe)
            ingredients = ingredients.output.ingredients
            ingredients_report = evaluate_list(df_ingredients, ingredients)
        except Exception as e:
            ingredients = DEFAULT_INGREDIENTS_VALUE
            ingredients_report = {"error": str(e)}
        ingredient_end = datetime.now()
        ingredient_time = ingredient_end - ingredient_start
        print("Ingredients Report:")
        print(ingredients_report)
        ingredients_path = os.path.join(base_dir, f"ingredient_extraction_results_{model_name}.json")
        save_report(ingredients_report, ingredients_path)


        # ==============================================
        # 6. ACTION EXTRACTION
        # ==============================================
        
        action_start = datetime.now()
        try:
            actions = await action_extractor.run(recipe)
            actions = actions.output.actions
            actions_report = evaluate_list([], actions)  # No ground truth for actions in this example
        except Exception as e:
            actions = DEFAULT_ACTIONS_VALUE
            actions_report = {"error": str(e)}
        action_end = datetime.now()
        action_time = action_end - action_start
        

        # ==============================================
        # 6. TITLE EXTRACTION
        # ==============================================
        title_start = datetime.now()
        try:
            title = await title_extractor.run()
            title = title.output.title
            most_similar_title = max(df_titles, key=lambda t: evaluate_extraction(t, title)['iou'])
            title_report = evaluate_extraction(most_similar_title, title)
        except Exception as e:
            title = DEFAULT_TITLE_VALUE
            title_report = {"error": str(e)}
            
        title_end = datetime.now()
        title_time = (title_end - title_start)

        print("Title Report:")
        print(title_report)


        title_path = os.path.join(base_dir, f"title_extraction_results_{model_name}.json")
        save_report(title_report, title_path)

        # ==============================================
        # 7. OUTPUT FORMATION
        # ==============================================

        new_recipe = Recipe(title, recipe, ingredients, [])  # actions are not extracted in this version
        new_row = pd.DataFrame([
        {
            "title": new_recipe.title,
            "text": new_recipe.text,
            "ingredients": new_recipe.ingredients,
            "actions": new_recipe.actions,
            "normalization_time": normalization_time.microseconds / 1_000_000,
            "validation_time": validation_time.microseconds / 1_000_000,
            "boundary_time": boundary_time.microseconds / 1_000_000,
            "ingredient_time": ingredient_time.microseconds / 1_000_000,
            "action_time": action_time.microseconds / 1_000_000,
            "title_time": title_time.microseconds / 1_000_000,
            "total_time": (
                normalization_time.microseconds + validation_time.microseconds + boundary_time.microseconds +
                ingredient_time.microseconds + action_time.microseconds + title_time.microseconds
            ) / 1_000_000,
            "text_length": len(recipe.split()),

        }])
        recipe_list.append(recipe)
        df_extracted = pd.concat([df_extracted, new_row], ignore_index=True)

    return df_extracted


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Process recipe files")
    parser.add_argument("--num_recipes", type=int, default=5, help="Number of recipes to process")
    parser.add_argument("--model_name", type=str, default="gepeto", help="Name of the model to use")
    args = parser.parse_args()

    num_recipes = args.num_recipes
    model_name = args.model_name
    print(f"Processing {num_recipes} files...")
    df_path = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/agentic/data/recipe1m_10000.parquet"
    df = pd.read_parquet(df_path)
    df = df[10:10+num_recipes]
    """
    
        Unnamed: 0	title	ingredients	directions	link	source	NER
    0	0	No-Bake Nut Cookies	["1 c. firmly packed brown sugar", "1/2 c. eva...	In a heavy 2-quart saucepan, mix brown sugar, ...	www.cookbooks.com/Recipe-Details.aspx?id=44874	Gathered	["brown sugar", "milk", "vanilla", "nuts", "bu...
    1	1	Jewell Ball'S Chicken	["1 small jar chipped beef, cut up", "4 boned ...	Place chipped beef on bottom of baking dish. P...	www.cookbooks.com/Recipe-Details.aspx?id=699419	Gathered	["beef", "chicken breasts", "cream of mushroom...
    2	2	Creamy Corn	["2 (16 oz.) pkg. frozen corn", "1 (8 oz.) pkg...	In a slow cooker, combine all ingredients. Cov...	www.cookbooks.com/Recipe-Details.aspx?id=10570	Gathered	["frozen corn", "cream cheese", "butter", "gar...
    3	3	Chicken Funny	["1 large whole chicken", "2 (10 1/2 oz.) cans...	Boil and debone chicken. Put bite size pieces ...	www.cookbooks.com/Recipe-Details.aspx?id=897570	Gathered	["chicken", "chicken gravy", "cream of mushroo...
    4	4	Reeses Cups(Candy)	["1 c. peanut butter", "3/4 c. graham cracker ...	Combine first four ingredients and press in 13...	www.cookbooks.com/Recipe-Details.aspx?id=659239	Gathered	["peanut butter", "graham cracker crumbs", "bu..
        
    """
    base_dir = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/agentic/results"



    recipes = asyncio.run(main(df))
    title_path = os.path.join(base_dir, f"title_extraction_results_{model_name}-{num_recipes}.json")

    recipes.to_csv("recipes.csv")

    print("\n")
    print("="*100)
    print(f"\n RESULTS\n")
    print("="*100)
    print("\n")
    print(recipes)
    
    