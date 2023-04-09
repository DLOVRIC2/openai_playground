import requests
import os
from dotenv import load_dotenv

load_dotenv()

# voice settings
STABILITY = 0.1
SIMILARITY_BOOST = 0.8

# streaming chunk size
CHUNK_SIZE = 1024

XI_API_KEY = os.getenv("ELEVEN_API_KEY")
OUTPUT_PATH = os.path.abspath("./output_voice.mp3")

headers = {
  "Accept": "application/json",
  "xi-api-key": XI_API_KEY,
}


data = {
  "text": "Some very long text to be read by the voice",
  "voice_settings": {
    "stability": STABILITY,
    "similarity_boost": SIMILARITY_BOOST
  }
}

voice_id = "21m00Tcm4TlvDq8ikWAM"
tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

response = requests.post(tts_url, json=data, headers=headers, stream=True)

with open(OUTPUT_PATH, 'wb') as f:
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            f.write(chunk)

# Retrieve history. It should contain generated sample.
history_url = "https://api.elevenlabs.io/v1/history"

headers = {
  "Accept": "application/json",
  "xi-api-key": XI_API_KEY
}

response = requests.get(history_url, headers=headers)

print(response.text)