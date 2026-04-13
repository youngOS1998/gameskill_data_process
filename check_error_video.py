import json

bad_lines = []
with open("./gameskill_merged_train.jsonl", "r") as f:
    for i, line in enumerate(f, start=1):
        sample = json.loads(line)
        v = sample.get("video", None)
        if v is None:
            continue
        if isinstance(v, str):
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            continue
        bad_lines.append((i, type(v), v))

print("Bad video entries:", bad_lines[:50])