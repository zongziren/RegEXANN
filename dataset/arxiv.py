from datasets import load_dataset
import numpy as np

print("Downloading dataset...")
ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=False)

titles = []
vecs = []

for i, item in enumerate(ds):
    if i >= 100000:
        break
    titles.append(item["title"].replace("\n", " "))
    vecs.append(item["vector"])

print("Saving titles to arxiv_titles_100k.txt ...")
with open("arxiv_titles_100k.txt", "w", encoding="utf-8") as f:
    for title in titles:
        f.write(title + "\n")


print("Saving vectors to arxiv_vectors_100k.npy ...")
vecs_np = np.array(vecs, dtype=np.float32)
np.save("arxiv_vectors_100k.npy", vecs_np)


def save_fvecs(filename, array):
    n, d = array.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(array[i].astype(np.float32).tobytes())

print("Saving vectors to arxiv_vectors_100k.fvecs ...")
save_fvecs("arxiv_vectors_100k.fvecs", vecs_np)

print(" Done: Titles, .npy and .fvecs saved.")
