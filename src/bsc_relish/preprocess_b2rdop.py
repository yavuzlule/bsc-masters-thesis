import json
import pandas as pd
import numpy as np
import os 


import json
import pandas as pd
from pathlib import Path


from bsc_relish.preprocess import (
    run_preprocessing,
    save_dataset,
    split_text_into_chunks
)


def process_json_file(json_data, df):
    """
    Process a single JSON file and append its data to the DataFrame.
    Extracts identifier as title, genre as category, and concatenated paragraphs as text.
    
    Args:
        json_data (dict): Dictionary containing the JSON data
        df (pd.DataFrame): DataFrame to append the row to
    
    Returns:
        pd.DataFrame: Updated DataFrame with the new row appended
    """
    # Extract the required fields from JSON,
    json_data = json_data[0]
    title = json_data.get('identifier', '')
    category = json_data.get('sifter_metadata', {}).get('Genre', '')
    language = json_data.get('sifter_metadata', {}).get('Language', '')
    
    # Concatenate paragraphs with newlines
    paragraphs = json_data.get('content', {}).get('paragraphs', [])
    text = '\n'.join(paragraphs) if isinstance(paragraphs, list) else ''
    label = 1 if category.lower() == 'cooking' or category.lower() == 'cookery' else 1
    new_row = {
        'title': title,
        'text': text,
        'category': category,
        'language': language,
        'label': label
        
    }
    
    # Append the row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df


def process_folders(base_folder, df=None):
    """
    Iterate over subfolders in the base folder and process all JSON files.
    
    Args:
        base_folder (str): Path to the base folder
        df (pd.DataFrame, optional): DataFrame to append to. If None, creates a new one.
    
    Returns:
        pd.DataFrame: DataFrame containing all processed JSON data
    """
    # Initialize DataFrame if not provided
    if df is None:
        df = pd.DataFrame(columns=['title', 'text', 'category'])
    
    # Get all subfolders
    base_path = Path(base_folder)
    
    if not base_path.exists():
        raise ValueError(f"Base folder does not exist: {base_folder}")
    
    # Iterate through subfolders
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            print(f"Processing subfolder: {subfolder.name}")
            
            # Iterate through JSON files in the subfolder
            for json_file in subfolder.glob('*.json'):
                try:
                    # Read the JSON file
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Process the JSON file
                    df = process_json_file(json_data, df)
                    print(f"  Processed: {json_file.name}")
                
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON in {json_file.name}: {e}")
                except Exception as e:
                    print(f"  Error processing {json_file.name}: {e}")
    
    return df

"""

 "columns": [
    "label",
    "file_name",
    "file_path",
    "parent_folder",
    "chunk_text",
    "chunk_index",
    "embedding_model",
    "embedding"
  ],

  
     if config["preprocessing"]["chunking"]["enabled"]:
        df = split_text_into_chunks(
            df,
            text_column="text",
            max_words=config["preprocessing"]["chunking"]["max_words"],
        )
        df.drop(columns=["text"], inplace=True)

        text_column = "chunk_text"
    else:
        text_column = "text"

"""

def main():
    result_df = process_folders("/Users/yavuzlule/Desktop/bsc-relish/data/external/internet_archive")
    result_df = result_df[:5]
    df = split_text_into_chunks(
            result_df,
            text_column="text",
            max_words=256,
        )
    df.drop(columns=["text"], inplace=True)
    df.to_parquet("/Users/yavuzlule/Desktop/bsc-relish/data/interim/b2drop_dataset.parquet", index=False, compression='snappy')
    print(df.head())

if __name__ == "__main__":
    main()
