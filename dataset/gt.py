import re
import numpy as np
from typing import List
from tqdm import tqdm

# ===== 1. 加载 .fvecs 文件 =====
def read_fvecs(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        data = f.read()
    dim = int.from_bytes(data[:4], byteorder="little")
    total = len(data) // ((dim + 1) * 4)
    arr = np.frombuffer(data, dtype=np.float32).reshape(total, dim + 1)
    return arr[:, 1:]

# ===== 2. 文本清洗（保留字母和空格） =====
def clean_title(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===== 3. 读取标题文件 =====
def load_titles(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [clean_title(line.strip()) for line in f]

# ===== 4. 读取查询文件 =====
def load_query_file(path: str) -> List[tuple]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            regex = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            queries.append((regex, vector))
    return queries

# ===== 5. 余弦相似度计算 =====
def cosine_similarity(q: np.ndarray, db: np.ndarray):
    q_norm = np.linalg.norm(q)
    db_norm = np.linalg.norm(db, axis=1)
    sim = db @ q / (q_norm * db_norm + 1e-8)
    return sim

# ===== 6. 正则过滤 + 向量相似度 + TopK =====
def search_groundtruth(regex: str, query_vec: np.ndarray, titles: List[str], vectors: np.ndarray, topk=10) -> List[int]:
    pattern = re.compile(regex)
    matched_ids = [i for i, t in enumerate(titles) if pattern.search(t)]
    if not matched_ids:
        return []
    sub_vectors = vectors[matched_ids]
    sims = cosine_similarity(query_vec, sub_vectors)
    topk_idx = np.argsort(-sims)[:topk]
    return [matched_ids[i] for i in topk_idx]

# ===== 7. 主程序入口 =====
if __name__ == "__main__":
    vectors = read_fvecs("/home/shurui/RFVEC/RegEXANN/dataset/SIFT100/sift100.fvecs")            # N x d
    titles = load_titles("/home/shurui/RFVEC/RegEXANN/dataset/SIFT100/sift100_titles.txt")                # N 行文本
    queries = load_query_file("/home/shurui/RFVEC/RegEXANN/dataset/SIFT100/query.txt")            # M 个查询
    topK = 10

    with open("/home/shurui/RFVEC/RegEXANN/dataset/SIFT100/groundtruth.txt", "w") as fout:
        for regex, qvec in tqdm(queries):
            top_ids = search_groundtruth(regex, qvec, titles, vectors, topk=topK)
            fout.write(" ".join(map(str, top_ids)) + "\n")

    print("✅ 所有 groundtruth 查询已完成，写入 groundtruth.txt")
