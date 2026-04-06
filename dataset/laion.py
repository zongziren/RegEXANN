from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def write_fvecs(path, data):
    data = np.array(data, dtype=np.float32)
    dim = data.shape[1]
    with open(path, "wb") as f:
        for vec in data:
            f.write(np.array([dim], dtype=np.int32).tobytes())
            f.write(vec.tobytes())

def export_laion_face(n_samples=1000000, output_vec="laion_text_1M.fvecs", output_txt="laion_text_1M.txt"):
    print(f"[INFO] Loading streaming dataset...")
    ds = load_dataset("FacePerceiver/laion-face", split="train", streaming=True)

    print(f"[INFO] Sampling {n_samples} items...")
    texts = []
    vectors = []

    for item in tqdm(ds.shuffle(buffer_size=10000).take(n_samples), total=n_samples):
        if item.get("text") and item.get("text_embedding"):
            texts.append(item["text"].replace("\n", " "))  # Clean newline
            vectors.append(item["text_embedding"])

    print(f"[INFO] Saving vectors to {output_vec}")
    write_fvecs(output_vec, vectors)

    print(f"[INFO] Saving text to {output_txt}")
    with open(output_txt, "w", encoding="utf-8") as fout:
        for line in texts:
            fout.write(line + "\n")

    print("✅ Export completed.")

# 运行
if __name__ == "__main__":
    export_laion_face(n_samples=1000000)
