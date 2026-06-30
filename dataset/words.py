import struct
import re
import numpy as np
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# Config
# =========================
LANG = "en"
BASE_N = 8000

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 128

BASE_WORD_TXT = "base_words.txt"
BASE_FVECS = "base.fvecs"
BASE_NPY = "base_vectors.npy"


def write_txt(path, words):
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")


def write_fvecs(path, arr):
    """
    fvecs format:
    [int32 dim][float32 * dim]
    [int32 dim][float32 * dim]
    ...
    """
    arr = arr.astype(np.float32)

    with open(path, "wb") as f:
        for vec in arr:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(vec.tobytes())


def main():
    print("Generating word list from wordfreq...")

    base_words = top_n_list(LANG, BASE_N)
    # wordfreq's English list is already letters-only in practice, but run
    # through the shared clean_text() for consistency with the other
    # datasets (no filtering afterwards, to keep the count == BASE_N).
    base_words = [clean_text(w) for w in base_words]

    print("base words:", len(base_words))

    write_txt(BASE_WORD_TXT, base_words)

    print("Saved text file:")
    print(" ", BASE_WORD_TXT)

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding base words...")
    base_vecs = model.encode(
        base_words,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    print("base vector shape:", base_vecs.shape)

    assert base_vecs.shape == (BASE_N, 768), base_vecs.shape

    np.save(BASE_NPY, base_vecs)
    write_fvecs(BASE_FVECS, base_vecs)

    print("Saved vector files:")
    print(" ", BASE_NPY)
    print(" ", BASE_FVECS)

    print("Done.")


if __name__ == "__main__":
    main()
