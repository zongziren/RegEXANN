import numpy as np
import re
import random
from tqdm import tqdm
import argparse

def read_fvecs(filename):
    with open(filename, "rb") as f:
        dim = int.from_bytes(f.read(4), byteorder="little")
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:]

def simple_substring(title: str) -> str:
    words = re.findall(r'[a-zA-Z]{3,}', title)  # 提取连续英文词，至少3个字母
    max_len = 50

    # 尝试从头开始拼接连续单词
    for start in range(len(words)):
        phrase = words[start]
        for end in range(start + 1, len(words)):
            candidate = " ".join(words[start:end + 1])
            if len(candidate) > max_len:
                break
            phrase = candidate
        if len(phrase) <= max_len:
            return phrase

    return "data"  # fallback

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

            fq.write(f"{substring} {' '.join(f'{x:.6f}' for x in query_vec)}\n")

            candidates = [i for i, title in enumerate(titles) if pattern.search(title)]
            dists = [(i, euclidean(query_vec, vectors[i])) for i in candidates]
            top_k = sorted(dists, key=lambda x: x[1])[:topk]

            fg.write(" ".join(str(i) for i, _ in top_k) + "\n")

    print(f"Saved queries to {query_out} and groundtruth to {gt_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries and groundtruth")
    parser.add_argument("title_file", help="Path to cleaned title .txt file")
    parser.add_argument("fvec_file", help="Path to .fvecs vector file")
    parser.add_argument("query_out", help="Path to output query file")
    parser.add_argument("gt_out", help="Path to output groundtruth file")
    parser.add_argument("--n_queries", type=int, default=100, help="Number of queries to generate")
    parser.add_argument("--topk", type=int, default=10, help="Top-K groundtruth results per query")
    args = parser.parse_args()

    generate_queries_and_groundtruth(
        args.title_file,
        args.fvec_file,
        args.query_out,
        args.gt_out,
        n_queries=args.n_queries,
        topk=args.topk
    )


#python gen_query.py ./sift100K/sift_titles_100k_clean.txt ./sift100K/sift_vectors_100k.fvecs ./sift100K/query.txt ./sift100K/groundtruth.txt --n_queries 1000 --topk 10
#python gen_query.py ./sift/sift_titles_clean.txt ./sift/sift_vectors.fvecs ./sift/query.txt ./sift/groundtruth.txt --n_queries 10 --topk 10