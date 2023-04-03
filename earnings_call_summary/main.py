import os
from dotenv import load_dotenv
import openai
import requests
import numpy as np

# Load the environment variables from the .env file
load_dotenv()
openai.api_key = os.getenv("API_KEY")

# Preparing the text to pass into the chat gpt
url = "https://gist.githubusercontent.com/hackingthemarkets/e664894b65b31cbe8993e02d25d26768/raw/618afe09d07979cc72911ce79634ab5d2cc19a54/nvidia-earnings-call.txt"
prompt_text = requests.get(url)
transcript = prompt_text.text

# Splitting the text into chunks
words = transcript.split(" ")
chunks = np.array_split(words, 6)

summary_response = []
for chunk in chunks:

    senteces = " ".join(list(chunk))

    prompt= f"{senteces}\n\ntl;dr:"

    # Calling chat gpt
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.3,
      max_tokens=140,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=1,
    )

    response_text = response["choices"][0]["text"]
    summary_response.append(response_text)

full_summary = "".join(summary_response)
