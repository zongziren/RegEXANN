import argparse
import csv
import time
import numpy as np
import psycopg2
from tqdm import tqdm


def read_fvecs_dim(path):
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    return int(data[0])


def read_query_file(path, expected_dim):
    """
    Query format:
        regex v1 v2 ... vd
    """
    patterns = []
    qvecs = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < expected_dim + 1:
                raise ValueError(
                    f"Line {line_no}: too few columns, got {len(parts)}, expected {expected_dim + 1}"
                )

            pattern = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            if len(vec) != expected_dim:
                raise ValueError(
                    f"Line {line_no}: query dim {len(vec)} != expected {expected_dim}"
                )

            patterns.append(pattern)
            qvecs.append(vec)

    return patterns, qvecs


def read_groundtruth(path):
    gt = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ids = []
            for x in line.replace(",", " ").split():
                try:
                    ids.append(int(float(x)))
                except ValueError:
                    pass

            gt.append(ids)

    return gt


def vec_to_pg(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def recall_at_k(results, groundtruth, k):
    n = min(len(results), len(groundtruth))
    if n == 0:
        return 0.0

    total = 0.0
    valid = 0

    for i in range(n):
        pred = results[i][:k]
        gt = groundtruth[i][:k]

        if not gt:
            continue

        total += len(set(pred) & set(gt)) / min(k, len(gt))
        valid += 1

    if valid == 0:
        return 0.0

    return 100.0 * total / valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="regexann_pg")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--vecs", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dim = read_fvecs_dim(args.vecs)
    patterns, qvecs = read_query_file(args.query, dim)
    gt = read_groundtruth(args.groundtruth)

    n = min(len(patterns), len(qvecs), len(gt))

    print(f"VECTOR_DIM,{dim}", flush=True)
    print(f"NUM_PATTERNS,{len(patterns)}", flush=True)
    print(f"NUM_QVECS,{len(qvecs)}", flush=True)
    print(f"NUM_GT,{len(gt)}", flush=True)
    print(f"NUM_USED,{n}", flush=True)

    conn = psycopg2.connect(
        dbname=args.db,
        host=args.host,
        port=args.port
    )
    cur = conn.cursor()

    all_results = []
    latencies = []

    total_start = time.perf_counter()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "regex", "latency_ms", "result_ids"])

        for qid in tqdm(range(n)):
            regex = patterns[qid]
            qvec = vec_to_pg(qvecs[qid])

            sql = """
                SELECT id
                FROM items
                WHERE text ~* %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """

            start = time.perf_counter()
            cur.execute(sql, (regex, qvec, args.k))
            rows = cur.fetchall()
            latency_ms = (time.perf_counter() - start) * 1000.0

            ids = [r[0] for r in rows]

            all_results.append(ids)
            latencies.append(latency_ms)

            writer.writerow([
                qid,
                regex,
                f"{latency_ms:.6f}",
                " ".join(map(str, ids))
            ])

    total_time = time.perf_counter() - total_start

    recall = recall_at_k(all_results, gt, args.k)
    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    qps = n / total_time if total_time > 0 else 0.0

    print(f"RECALL_PCT,{recall:.3f}", flush=True)
    print(f"AVG_TIME_MS,{avg_ms:.3f}", flush=True)
    print(f"QPS,{qps:.3f}", flush=True)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
