import json, pathlib, random, re

SRC = pathlib.Path("data.jsonl")
OUT_TRAIN = pathlib.Path("data/train.jsonl")
OUT_DEV   = pathlib.Path("data/dev.jsonl")
OUT_TRAIN.parent.mkdir(exist_ok=True)

rows=[]
for line in SRC.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    try:
        obj = json.loads(line)
        p = obj.get("prompt","").strip()
        y = obj.get("completion","").strip()
        if p and y:
            rows.append({"prompt": p, "completion": y})
    except Exception:
        pass

# stratified by label text
from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[r["completion"].strip()].append(r)

random.seed(42)
train, dev = [], []
for lab, items in by.items():
    random.shuffle(items)
    k = max(1, int(0.8*len(items)))  # 80/20 split (small dev)
    train += items[:k]
    dev += items[k:]

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")

write_jsonl(OUT_TRAIN, train)
write_jsonl(OUT_DEV, dev)
print("train:", len(train), "dev:", len(dev))
