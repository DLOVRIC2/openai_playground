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
STABILITY = 0.1
SIMILARITY_BOOST = 0.8

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

# # prepare Q&A embeddings dataframe
# question_df = pd.read_csv('data/questions_with_embeddings.csv')
# question_df['embedding'] = question_df['embedding'].apply(eval).apply(np.array)
#

def transcribe_message(audio):

    global messages, question_df

    # API now requires an extension so we will rename the file
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    # Transcribing the audio input
    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # # Converting the transcript to vectors format
    # question_vector = get_embedding(transcript["text"], engine="text-embedding-ada-002")
    # question_df["similarities"] = question_df.apply(lambda x: cosine_similarity(x, question_vector))
    # question_df = question_df.sort_values("similarities", ascending=False)
    #
    # best_answer = question_df.iloc[0]["answer"]
    #
    # # If we want to use a custom vectorised df to store answers.
    # user_text_with_embedding = f"Using the following text, answer the question '{transcript['text']}'. " \
    #                            f"{config.ADVISOR_CUSTOM_PROMPT}: {best_answer}"
    user_text_no_embedding = f"{config.ADVISOR_CUSTOM_PROMPT} {config.ADVISOR_STLYE}. " \
                             f"With that in mind, answer the following question: \n{transcript['text']}"

    messages.append({"role": "user", "content": user_text_no_embedding})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.MORGAN_FREEMAN}/stream"
    data = {
        "text": system_message["content"].replace('"', ''),  # Replacing " to avoid confusion in audio
        "voice_settings": {
            "stability": STABILITY,
            "similarity_boost": SIMILARITY_BOOST
        }
    }

    response = requests.post(tts_url, json=data, headers=headers, stream=True)

    output_filename = "reply.mp3"
    with open(output_filename, "wb") as output:
        output.write(response.content)

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript, output_filename


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

