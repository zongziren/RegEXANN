import os
import re
from datasets import load_dataset
import random
import numpy as np


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


MSONG_FVECS = "./msong/msong_base.fvecs"
AUDIO_FVECS = "./audio/audio_base.fvecs"

OUT_MSONG = "./msong/msong_titles.txt"
OUT_AUDIO = "./audio/audio_titles.txt"
random.seed(42)
random.seed(42)

def count_fvecs(fname):
    file_size = os.path.getsize(fname)

    with open(fname, "rb") as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]

    record_size = 4 + dim * 4
    n = file_size // record_size

    if file_size % record_size != 0:
        raise ValueError(f"Invalid fvecs file: {fname}")

    return n, dim


n_msong, d_msong = count_fvecs(MSONG_FVECS)
n_audio, d_audio = count_fvecs(AUDIO_FVECS)

print("msong:", n_msong, d_msong)
print("audio:", n_audio, d_audio)

print("Loading Spotify titles...")
ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")

titles = []

for item in ds:
    name = clean_text(item.get("track_name", ""))
    artist = clean_text(item.get("artists", ""))

    if not name or name.lower() == "nan":
        continue

    if artist and artist.lower() != "nan":
        title = f"{name} - {artist}"
    else:
        title = name

    titles.append(title)

titles = list(dict.fromkeys(titles))

print("unique source titles:", len(titles))

if len(titles) == 0:
    raise RuntimeError("No titles loaded from Spotify dataset")


def write_random_titles(out_file, n):
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(random.choice(titles) + "\n")

            if (i + 1) % 100000 == 0:
                print(f"{out_file}: written {i + 1}/{n}")


print("Generating msong titles...")
write_random_titles(OUT_MSONG, n_msong)

print("Generating audio titles...")
write_random_titles(OUT_AUDIO, n_audio)

print("Done.")
print(f"{OUT_MSONG}: {n_msong} lines")
print(f"{OUT_AUDIO}: {n_audio} lines")