from flask import Flask, render_template, request
import json
import openai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity

load_dotenv()
openai.api_key = os.getenv("API_KEY")

df = pd.read_csv("G:\My Drive\Arc Capital\Python\openai_playground\word_embeddings\word_embeddings.csv")

app = Flask(__name__)

# Load data for search
with open('data.json') as f:
    data = json.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    search_term = ""
    results = []
    if request.method == 'POST':
        search_term = request.form['search']
        results = search_data(search_term)
    return render_template('index.html', search_term=search_term, results=results)


def search_data(term):

    search_term_vector = get_embedding(term, engine="text-embedding-ada-002")

    df = pd.read_csv("G:\My Drive\Arc Capital\Python\openai_playground\word_embeddings\word_embeddings.csv")
    df["embedding"] = df["embedding"].apply(eval).apply(np.array)
    df["similarities"] = df["embedding"].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(5)
    results = sorted_by_similarity["text"].values.tolist()

    return results


if __name__ == '__main__':
    app.run(debug=True)