import gradio as gr
import openai
import os
import requests
from dotenv import load_dotenv
import soundfile as sf
import pyttsx3
import subprocess


load_dotenv()
openai.api_key = os.getenv("API_KEY")
XI_API_KEY = os.getenv("ELEVEN_API_KEY") # ElevenLabs

# voice settings
STABILITY = 0
SIMILARITY_BOOST = 0

# streaming chunk size
CHUNK_SIZE = 1024
OUTPUT_PATH = os.path.abspath("./output_voice.mp3")
voice_id = "21m00Tcm4TlvDq8ikWAM"  # American Female


# Headers for ElevenLabs (Voice output)
headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY,
    "Content-Type": "application/json"
}

# ChatGPT Prompt Engineering
messages = [{"role": "system", "content": 'You are a therapist. Respond to all input in 25 words or less.'}]


def transcribe_message(audio):
    global messages

    sample_rate, audio_data = audio

    file_name = "recorded_audio.wav"
    file_path = os.path.abspath(file_name)
    sf.write(file_path, audio_data, sample_rate)

    audio_file = open(file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    data = {
        "text": system_message["content"],
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

    # Play the audio file
    audio_path = r"G:\My Drive\Arc Capital\Python\openai_playground\audio_gpt\output_voice.mp3"
    os.system(f'start "" "{audio_path}"')

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript


ui = gr.Interface(fn=transcribe_message, inputs=gr.Audio(source="microphone", type="numpy", interactive=True), outputs="text")
ui.launch()
