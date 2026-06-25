import os
import re
import random
import struct
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


OUT_DIR = "dataset/dbpedia"
os.makedirs(OUT_DIR, exist_ok=True)

N_BASE = 1000000
N_QUERY = 1000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def clean_text(s: str) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def write_fvecs(path, vectors):
    with open(path, "wb") as f:
        for v in vectors:
            v = np.asarray(v, dtype=np.float32)
            f.write(struct.pack("i", len(v)))
            f.write(v.tobytes())


print("Loading DBpedia from HuggingFace...")
ds = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True)

base_vec_path = os.path.join(OUT_DIR, "vectors.fvecs")
base_str_path = os.path.join(OUT_DIR, "strings.txt")
query_path = os.path.join(OUT_DIR, "query.txt")
query_vec_path = os.path.join(OUT_DIR, "query_vectors.fvecs")

base_count = 0
query_count = 0

with open(base_vec_path, "wb") as fvec, \
     open(base_str_path, "w", encoding="utf-8") as fstr, \
     open(query_path, "w", encoding="utf-8") as fq, \
     open(query_vec_path, "wb") as fqvec:

    for row in tqdm(ds, total=N_BASE + N_QUERY):
        title = clean_text(row.get("title", ""))
        text = clean_text(row.get("text", ""))
        string = clean_text(title + " " + text)

        emb = row.get("openai", None)
        if emb is None:
            emb = row.get("embedding", None)
        if emb is None:
            raise KeyError(f"Cannot find embedding field. Available keys: {row.keys()}")

        v = np.asarray(emb, dtype=np.float32)
        if v.shape[0] != 1536:
            raise ValueError(f"Unexpected dim: {v.shape[0]}")

        if base_count < N_BASE:
            fvec.write(struct.pack("i", len(v)))
            fvec.write(v.tobytes())
            fstr.write(string + "\n")
            base_count += 1
            continue

        if query_count < N_QUERY:
            fqvec.write(struct.pack("i", len(v)))
            fqvec.write(v.tobytes())

            patterns = [
                ".*university.*",
                ".*city.*",
                ".*film.*",
                ".*football.*",
                ".*American.*",
                ".*born.*",
                ".*river.*",
                ".*album.*",
                ".*company.*",
                ".*school.*",
            ]
            fq.write(patterns[query_count % len(patterns)] + "\n")
            query_count += 1

        if base_count >= N_BASE and query_count >= N_QUERY:
            break

print("Done.")
print(f"Base vectors : {base_count} -> {base_vec_path}")
print(f"Base strings : {base_count} -> {base_str_path}")
print(f"Query regex  : {query_count} -> {query_path}")
print(f"Query vectors: {query_count} -> {query_vec_path}")