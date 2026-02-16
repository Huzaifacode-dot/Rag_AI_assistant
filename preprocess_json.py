#this read chunks
from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import pandas as pd
import joblib

# Load model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text_list):
    return model.encode(text_list)

jsons = os.listdir("merged_jsons")
print("Files found:", jsons)

my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"merged_jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating embedding for {json_file}")

    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content["chunks"]):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i].tolist()  # safer for saving
        chunk_id += 1
        my_dicts.append(chunk)

    # break  # remove later when testing multiple files

df = pd.DataFrame.from_records(my_dicts)

print("Saving file now...")
print("Saving to:", os.getcwd())

joblib.dump(df, "embedding.joblib")

print("Done!")
