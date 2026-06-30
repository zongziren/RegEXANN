import os
import re
import numpy as np
import lance


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Streamed directly from the Hugging Face Hub via the hf:// URI — lance
# supports this natively (pylance + huggingface_hub installed), so no
# separate download step is needed for LAION-1M.
# Source: https://huggingface.co/datasets/lance-format/laion-1m
DATA_PATH = "hf://datasets/lance-format/laion-1m/data/train.lance"

OUT_FVECS = "laion_base.fvecs"
OUT_TEXT = "laion_captions.txt"

TARGET_N = 1_000_000
VECTOR_FIELD = "img_emb"
TEXT_CANDIDATES = ["caption", "text", "TEXT", "title", "description"]


def save_fvec(f, vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    d = int(vec.shape[0])
    f.write(np.int32(d).tobytes())
    f.write(vec.tobytes())
    return d


def main():
    ds = lance.dataset(DATA_PATH)
    fields = ds.schema.names

    print("fields:", fields)
    print("rows:", ds.count_rows())

    if VECTOR_FIELD not in fields:
        raise ValueError(f"Cannot find vector field {VECTOR_FIELD}. Fields: {fields}")

    text_field = None
    for c in TEXT_CANDIDATES:
        if c in fields:
            text_field = c
            break

    if text_field is None:
        raise ValueError(f"Cannot find text field. Fields: {fields}")

    print("vector field:", VECTOR_FIELD)
    print("text field:", text_field)

    n = 0
    dim = None

    with open(OUT_FVECS, "wb") as vf, open(OUT_TEXT, "w", encoding="utf-8") as tf:
        for batch in ds.to_batches(columns=[VECTOR_FIELD, text_field], batch_size=2048):
            data = batch.to_pydict()
            vecs = data[VECTOR_FIELD]
            texts = data[text_field]

            for vec, txt in zip(vecs, texts):
                txt = clean_text(txt)
                if not txt:
                    continue

                d = save_fvec(vf, vec)

                if dim is None:
                    dim = d
                    print("dim:", dim)
                elif d != dim:
                    raise ValueError(f"dim mismatch: {d} != {dim}")

                tf.write(txt + "\n")
                n += 1

                if n % 100000 == 0:
                    print(f"written {n}/{TARGET_N}")

                if n >= TARGET_N:
                    break

            if n >= TARGET_N:
                break

    print("Done.")
    print("vectors:", n)
    print("dim:", dim)
    print("fvecs:", OUT_FVECS)
    print("text:", OUT_TEXT)


if __name__ == "__main__":
    main()