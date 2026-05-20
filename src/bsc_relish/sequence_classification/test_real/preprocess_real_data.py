import pandas as pd



def chunk_df(df, column_name):
    MAX_WORDS = 256

    rows = []
    for _, row in df.iterrows():
        text = row[column_name]

        # Split into words
        words = text.split()

        # Chunk into groups of 256 words
        for chunk_index, start in enumerate(range(0, len(words), MAX_WORDS)):
            chunk_words = words[start:start + MAX_WORDS]

            # Join back into text
            chunk_text = " ".join(chunk_words)

            # Copy original row
            new_row = row.copy()

            # Replace text with chunk
            new_row["recipe_text"] = chunk_text

            # Add chunk index
            new_row["chunk_index"] = chunk_index

            rows.append(new_row)

    # New dataframe
    chunked_df = pd.DataFrame(rows)

    return chunked_df





# Example: list of JSON objects (Python dicts)
data = pd.read_json("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/test_real/data/relish_dataset.json")

# Convert to DataFrame
df = pd.DataFrame(data)


# Save as Parquet
df.to_parquet("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/test_real/data/relish_dataset.parquet", index=False)

print(df.columns)

df.to_parquet("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/test_real/data/relish_dataset.parquet", index=False)