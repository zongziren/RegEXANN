import re
from datasets import load_dataset
import numpy as np


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


DATASET_NAME = "Qdrant/arxiv-titles-instructorxl-embeddings"
OUT_TITLE = "arxiv_titles_all.txt"
OUT_FVECS = "arxiv_vectors_all.fvecs"

print("Downloading dataset...")
ds = load_dataset(DATASET_NAME, split="train", streaming=False)

num_rows = len(ds)
print(f"Total rows: {num_rows}")

first_vec = np.asarray(ds[0]["vector"], dtype=np.float32)
dim = first_vec.shape[0]
print(f"Vector dimension: {dim}")

with open(OUT_TITLE, "w", encoding="utf-8") as title_f, \
     open(OUT_FVECS, "wb") as fvecs_f:

    for i, item in enumerate(ds):
        title = clean_text(item["title"])
        vec = np.asarray(item["vector"], dtype=np.float32)

        title_f.write(title + "\n")

        fvecs_f.write(np.int32(dim).tobytes())
        fvecs_f.write(vec.tobytes())

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{num_rows}")

print("Done.")
print(f"Titles saved to: {OUT_TITLE}")
print(f"Fvecs saved to: {OUT_FVECS}")
print(f"Total vectors: {num_rows}")
print(f"Dimension: {dim}")