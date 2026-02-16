# How to use this RAG based AI assistant on your own data
## Step 1 - Collect the audios of all the videos you want (for example some youtube playlist)
python -m yt_dlp -x --audio-format mp3 --playlist-items --ignore-errors --no-warnings -o "audio/%(playlist_index)s_%(title)s.%(ext)s" "youtube-video-url"

## Step 2 - Convert all mp3 to json
convert all the mp3 files to json by running mp3_to_json

## Step 3 - Convert the json files to vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 4 - Prompt generation and feeding to LLM
 Read the joblib file and load it into the memory.Then create a relevant prompt as per the user query and feed it to the LLM

 