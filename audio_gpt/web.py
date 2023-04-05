import gradio as gr
import openai
import os
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()
openai.api_key = os.getenv("API_KEY")


def transcribe_message(audio):

    sample_rate, audio_data = audio

    file_name = "recorded_audio.wav"
    file_path = os.path.abspath(file_name)
    sf.write(file_path, audio_data, sample_rate)

    audio_file = open(file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript["text"]


ui = gr.Interface(fn=transcribe_message, inputs=gr.Audio(source="microphone", type="numpy", interactive=True), outputs="text")
ui.launch()
