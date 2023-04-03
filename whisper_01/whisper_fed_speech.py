import whisper
from pytube import YouTube
import subprocess
import datetime
import ffmpeg

# Loading the base model with 74M parameters. Also available are: tiny, small,medium and large
model = whisper.load_model("base")

# Instantiating the YouTube object for the FED speech video
youtube_video_url = "https://www.youtube.com/watch?v=NT2H9iyd-ms"
youtube_video = YouTube(youtube_video_url)

# To check list of available methods in this class, uncomment the following code
# for i in dir(youtube_video):
#     print(i)

# We need to grab audio stream from the link (we need only audio at the moment)
streams = youtube_video.streams.filter(only_audio=True)

# We don't need the highest quality audio for this project, so we'll select the first audio stream available.
# If we want a higher quality transcription, we can select a higher quality audio stream and use a larger Whisper model
stream = streams.first()
# stream.download(filename="fed_meeting.mp4")

# We can do some additional processing on the audio file should we choose. I want to ignore any additional sound and
# speech after Jerome Powell speaks. So we'll use ffmpeg to do this. The command will start the audio file at the 375
# second mark where he starts with good afternoon, continue for 2715 seconds, and chop off the rest of the audio.
# The result will be saved in a new file called fed_meeting_trimmed.mp4.
input_file = r"G:\My Drive\Arc Capital\Python\openai_playground\fed_meeting.mp4"
# subprocess.run(['ffmpeg', '-ss', '378', '-i', input_file, '-t', '2715', 'fed_meeting_trimmed.mp4'], check=True)

# Transcription
t1 = datetime.datetime.now()
print(f"Started at {t1}")

# do the transcription
output = model.transcribe("fed_meeting_trimmed.mp4")


# Show time elapsed after transcription is complete
t2 = datetime.datetime.now()
print(f"ended at {t2}")
print(f"Time elapsed: {t2 - t1}")


with open("transcribed_text.txt", "w") as f:
    f.write(output["text"])

