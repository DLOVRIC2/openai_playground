from pytube import YouTube


url = {1: "https://www.youtube.com/watch?v=5cCW2a8kT3Q",
       2: "https://www.youtube.com/shorts/q0bK2HOxK-U",
       3: "https://www.youtube.com/shorts/TvH3sks8hQc",
       4: "https://youtu.be/zfrVau8cYD0"}
# Create a YouTube object with the URL
yt = YouTube(url[4])

# Get the first available audio stream
audio = yt.streams.filter(only_audio=True).first()

# Set the output file path and name
output_path = "./voices/"
output_file = f"{output_path}/{yt.title}.mp3"

audio.download(output_path=output_path, filename=yt.title)

# Rename the downloaded file to have an mp3 extension
import os
os.rename(f"{output_path}/{yt.title}", output_file)