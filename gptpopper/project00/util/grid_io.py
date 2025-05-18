import json, numpy as np
COLMAP = {0: "bg", 1: "cyan", 2: "red", 3: "blue", 4: "yellow", 5: "green"}
REV = {v: k for k, v in COLMAP.items()}

def load_pair(path: str):
    data = json.load(open(path))
    inp = np.array(data["train"][0]["input"], dtype=int)
    out = np.array(data["train"][0]["output"], dtype=int)
    return inp, out