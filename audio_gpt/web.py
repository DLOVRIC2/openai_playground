import gradio as gr
import openai
import os
import subprocess
from dotenv import load_dotenv
import soundfile as sf
import pyttsx3

load_dotenv()
openai.api_key = os.getenv("API_KEY")

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

    def say_message(message):
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()

    say_message(system_message["content"])

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript


ui = gr.Interface(fn=transcribe_message, inputs=gr.Audio(source="microphone", type="numpy", interactive=True), outputs="text")
ui.launch()
