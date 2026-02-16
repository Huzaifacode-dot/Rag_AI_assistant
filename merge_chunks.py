import os
import json
import math

n = 5  # number of chunks to merge

for file_name in os.listdir("jsons"):
    if file_name.endswith(".json"):
        file_path = os.path.join("jsons", file_name) 
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(data)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_groups = math.ceil(num_chunks/n)  # calculate number of groups

            for i in range(num_groups):
                start_index = i * n
                end_index = min((i + 1) * n, num_chunks)  # ensure we don't go out of bounds
                group_chunks = data["chunks"][start_index:end_index]

                merged_chunk = {
                    "number": group_chunks[0]["number"],
                    "title": group_chunks[0]["title"],
                    "start": group_chunks[0]["start"],
                    "end": group_chunks[-1]["end"],
                    "text": " ".join(chunk["text"] for chunk in group_chunks)
                }
                new_chunks.append(merged_chunk)
                os.makedirs("merged_jsons", exist_ok=True)  # create directory if it doesn't exist
                with open(f"merged_jsons/{file_name}", "w", encoding="utf-8") as f_out:
                    json.dump({"chunks": new_chunks, "text": data['text']}, f_out, indent=4)
                