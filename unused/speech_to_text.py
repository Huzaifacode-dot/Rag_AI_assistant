import whisper
import json

model = whisper.load_model("base")

result = model.transcribe(
    "audio/sample.mp3",
    language="hi",
    task="translate",
    word_timestamps=False   
)

print(result)
chunks = []
for segment in result["segments"]:
    chunks.append({"start":segment["start"],"end":segment["end"],"text":segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)