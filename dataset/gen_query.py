import numpy as np
import re
import random
from tqdm import tqdm

def read_fvecs(filename):
    with open(filename, "rb") as f:
        dim = int.from_bytes(f.read(4), byteorder="little")
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:]

def simple_substring(title: str) -> str:
    words = re.findall(r'[a-zA-Z]+', title)
    for w in words:
        if len(w) >= 3:
            return w[:min(10, len(w))]  # 截取前3~10个字符
    return "data"

def euclidean(a, b):
    return np.linalg.norm(a - b)

def generate_queries_and_groundtruth(
    title_file: str, fvec_file: str, query_out: str, gt_out: str, n_queries: int = 10, topk: int = 5):
    
    titles = [line.strip() for line in open(title_file, encoding='utf-8')]
    vectors = read_fvecs(fvec_file)
    assert len(titles) == len(vectors)

    indices = random.sample(range(len(titles)), n_queries)

    with open(query_out, "w", encoding="utf-8") as fq, open(gt_out, "w") as fg:
        for qid in tqdm(indices, desc="Generating queries"):
            query_vec = vectors[qid]
            substring = simple_substring(titles[qid])
            pattern = re.compile(re.escape(substring), re.IGNORECASE)

            # 写入查询
            fq.write(f"{substring} {' '.join(f'{x:.6f}' for x in query_vec)}\n")

            # 查找正则匹配的 candidates（纯子串）
            candidates = [i for i, title in enumerate(titles) if pattern.search(title)]
            dists = [(i, euclidean(query_vec, vectors[i])) for i in candidates]
            top_k = sorted(dists, key=lambda x: x[1])[:topk]

            # 写入 groundtruth
            fg.write(" ".join(str(i) for i, _ in top_k) + "\n")

    print(f"✅ Saved queries to {query_out} and groundtruth to {gt_out}")

# 示例调用
if __name__ == "__main__":
    generate_queries_and_groundtruth(
        "./arxiv100K/arxiv_titles_100k_clean.txt",
        "./arxiv100K/arxiv_vectors_100k.fvecs",
        "./arxiv100K/query.txt",
        "./arxiv100K/groundtruth.txt",
        n_queries=1000,
        topk=10
    )
