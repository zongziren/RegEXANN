import numpy as np

def read_fvecs(path):
    with open(path, "rb") as f:
        dim = int.from_bytes(f.read(4), byteorder="little")
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape(-1, dim + 1)

def write_fvecs(path, data):
    n, dim = data.shape
    with open(path, "wb") as f:
        for vec in data:
            f.write(np.array([dim], dtype=np.int32).tobytes())
            f.write(vec.astype(np.float32).tobytes())

def extract_first_n(input_path, output_path, n=100000):
    all_data = read_fvecs(input_path)
    print(f"Loaded {all_data.shape[0]} vectors of dimension {all_data.shape[1]}")
    subset = all_data[:n]
    write_fvecs(output_path, subset)
    print(f"Saved first {n} vectors to {output_path}")

# 示例调用
if __name__ == "__main__":
    extract_first_n(
        "./sift/sift_base.fvecs",
        "./sift/sift_base_100k.fvecs",
        n=100000
    )
