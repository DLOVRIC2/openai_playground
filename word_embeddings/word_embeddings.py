import openai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity


load_dotenv()
openai.api_key = os.getenv("API_KEY")

words = ["red", "potatoes", "soda", "cheese", "water", "blue", "crispy", "hamburger", "coffee", "green", "milk",
         "la croix", "yellow", "chocolate", "french fries", "latte", "cake", "brown", "cheeseburger", "espresso",
         "cheese cake", "black", "mocha", "fizzy", "carbon", "banana"]

# First time loading
# df = pd.DataFrame(words, columns=["text"])
# df["embedding"] = df["text"].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
# df.to_csv("word_embeddings.csv")

# Load the data
df = pd.read_csv("word_embeddings.csv")
df["embedding"] = df["embedding"].apply(eval).apply(np.array)
df2 = df.copy()

user_input = False
if user_input:
    search_term = input("Enter a search term: ") # i.e. Ice cream, hot dog, steak...
    search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
else:
    search_term = "Steak"
    search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")


df["similarities"] = df["embedding"].apply(lambda x: cosine_similarity(x, search_term_vector))
df.sort_values("similarities", ascending=False, inplace=True)
df.to_csv(f"results_for_{search_term}")

# You can also add vectors together
milk_vector = df["embedding"][10]
espresso_vector = df["embedding"][19]
milk_espresso_vector = milk_vector + espresso_vector

df2["similarities"] = df["embedding"].apply(lambda x: cosine_similarity(x, milk_espresso_vector))
df2.sort_values("similarities", ascending=False, inplace=True)
df2.to_csv(f"results_for_milk_espresso")

