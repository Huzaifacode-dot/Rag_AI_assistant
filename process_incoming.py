import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import json

# ✅ Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text_list):
    embedding = model.encode(text_list)
    return embedding

# ✅ Load stored embeddings
df = joblib.load("embedding.joblib")

print("API KEY:", os.getenv("rag_api"))

# ✅ Initialize Groq (use env variable recommended)
client = Groq(api_key=os.getenv("rag_api"))



incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])[0]

# find similarities of question embeddings with other embeddings
similarities = cosine_similarity(
    np.vstack(df['embedding'].values),
    [question_embedding]
).flatten()

top_results = 5
max_index = similarities.argsort()[::-1][:top_results]

new_df = df.loc[max_index]

# ---------- PROMPT ----------
prompt = f'''I am teaching DSA data structures and algorithms course for placement preparation.
Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text of the chunk.

Chunks:
{new_df[['title', 'number','start','end', 'text']].to_json(orient='records')}

-------------------------------------

User Question:
"{incoming_query}"

Please use ONLY the information in these chunks to answer the question.
Mention the video title and timestamp in your answer.
If you don't know the answer, say you don't know.
If user asks unrelated question, tell him that you can only answer questions related to the content of the videos.
'''

# Save prompt (as you had)
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# ---------- GROQ GENERATION ----------
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant answering questions strictly from provided context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

final_answer = response.choices[0].message.content

print("\nFinal Answer:\n")
print(final_answer)
