import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_data():
    return joblib.load("embedding.joblib")

model = load_embedding_model()
df = load_data()

client = Groq(api_key=os.getenv("rag_api"))

# ---------- FUNCTIONS ----------
def create_embedding(text):
    return model.encode([text])[0]

def retrieve_chunks(query, top_k=5):
    question_embedding = create_embedding(query)

    similarities = cosine_similarity(
        np.vstack(df['embedding'].values),
        [question_embedding]
    ).flatten()

    top_index = similarities.argsort()[::-1][:top_k]
    return df.loc[top_index]

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": "Answer only using the given context and mention video title and timestamp."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.title("🎓 Course Video RAG Assistant")

st.write("Ask questions about your course videos and get exact timestamps.")

user_query = st.text_input("Enter your question:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching videos..."):
            results = retrieve_chunks(user_query)

            prompt = f"""
Here are relevant video transcript chunks:

{results[['title','number','start','end','text']].to_json(orient='records')}

User Question:
{user_query}

Answer using only this information and mention video title and timestamp.
"""

            answer = generate_answer(prompt)

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("🔍 Retrieved Chunks")
        st.dataframe(results[['title','number','start','end']])
