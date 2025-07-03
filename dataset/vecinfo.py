import numpy as np

def fvecs_info(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        file_size = f.tell()

        bytes_per_vector = 4 + 4 * dim
        num_vectors = file_size // bytes_per_vector

    return num_vectors, dim

def read_fvecs(filename, num_vectors_to_read):
    vectors = []
    with open(filename, 'rb') as f:
        for _ in range(num_vectors_to_read):
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec = np.fromfile(f, dtype=np.float32, count=dim)
            vectors.append(vec)
    return np.stack(vectors)


filename = "./SIFT100/sift100.fvecs"
n, d = fvecs_info(filename)
print(f"Number of vectors: {n}")
print(f"Vector dimension: {d}")
print("\nFirst 10 vectors:")
first_10 = read_fvecs(filename, 3)
for i, vec in enumerate(first_10):
    print(f"Vector {i}: {vec}")