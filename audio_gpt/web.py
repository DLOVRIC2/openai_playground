import gradio as gr
import openai
import os
import requests
import config
from dotenv import load_dotenv
import soundfile as sf
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np

load_dotenv()
openai.api_key = os.getenv("API_KEY")
XI_API_KEY = os.getenv("ELEVEN_API_KEY") # ElevenLabs

# voice settings
STABILITY = 0
SIMILARITY_BOOST = 0

# streaming chunk size
CHUNK_SIZE = 1024
OUTPUT_PATH = os.path.abspath("./output_voice.mp3")
voice_id = config.MORGAN_FREEMAN


# Headers for ElevenLabs (Voice output)
headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY,
    "Content-Type": "application/json"
}

# ChatGPT Prompt Engineering
messages = [{"role": "system", "content": config.ADVISOR_CUSTOM_PROMPT}]

# prepare Q&A embeddings dataframe
question_df = pd.read_csv('data/questions_with_embeddings.csv')
question_df['embedding'] = question_df['embedding'].apply(eval).apply(np.array)


def transcribe_message(audio):

    global messages, question_df

    # Unpacking the audio file
    sample_rate, audio_data = audio
    file_name = "recorded_audio.wav"
    file_path = os.path.abspath(file_name)
    sf.write(file_path, audio_data, sample_rate)

    # Transcribing the audio input
    audio_file = open(file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Converting the transcript to vectors format
    question_vector = get_embedding(transcript["text"], engine="text-embedding-ada-002")
    question_df["similarities"] = question_df.apply(lambda x: cosine_similarity(x, question_vector))
    question_df = question_df.sort_values("similarities", ascending=False)

    best_answer = question_df.iloc[0]["answer"]

    # If we want to use a custom vectorised df to store answers.
    user_text_with_embedding = f"Using the following text, answer the question '{transcript['text']}'. " \
                               f"{config.ADVISOR_CUSTOM_PROMPT}: {best_answer}"
    user_text_no_embedding = f"{config.ADVISOR_CUSTOM_PROMPT} {config.ADVISOR_STLYE}. " \
                             f"With that in mind, answer the following question: \n{transcript['text']}"


    messages.append({"role": "user", "content": user_text_no_embedding})

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

    voice_id = "rHDmATXy0Mz2HBV1Hmah"
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

    # TODO: Get the audio file to be played by the gradio instaed of mp3 player
    # # Convert the MP3 file to a WAV file
    # audio_path = os.path.abspath("./output_voice.mp3")
    # wav_path = os.path.abspath("./output_voice.wav")
    # sound = AudioSegment.from_mp3(audio_path)
    # sound.export(wav_path, format="wav")
    #
    # # Read the audio file and return it along with the transcript
    # with open(wav_path, 'rb') as f:
    #     audio_bytes = f.read()
    # return chat_transcript, audio_bytes
    #

# ui = gr.Interface(fn=transcribe_message, inputs=gr.Audio(source="microphone", type="numpy", interactive=True), outputs=["text", gr.Audio])
ui = gr.Interface(fn=transcribe_message, inputs=gr.Audio(source="microphone", type="numpy", interactive=True), outputs="text")
ui.launch()

# set a custom theme
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)

with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input
    advisor = gr.Image(value=config.IMAGE).style(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe_message, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True)

