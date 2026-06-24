#!/usr/bin/env python3
import argparse
import csv
import time
import numpy as np
import psycopg2
from tqdm import tqdm


def read_query_txt(path):
    """
    Read query.txt.

    Supported formats:

    1) regex float float float ...
       deep.*learning 0.1 0.2 0.3 ...

    2) regex<TAB>float float float ...
       deep.*learning\t0.1 0.2 0.3 ...

    3) regex<TAB>[float,float,float,...]
       deep.*learning\t[0.1,0.2,0.3,...]

    Return:
        regexes: list[str]
        vectors: np.ndarray, shape = (nq, dim)
        dim: int
    """
    regexes = []
    vectors = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Prefer TAB split, because regex may contain spaces less often,
            # and query generators usually separate regex/vector by tab.
            if "\t" in line:
                regex, vec_str = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid query line {lineno}: expected regex + vector"
                    )
                regex, vec_str = parts

            regex = regex.strip()
            vec_str = vec_str.strip()

            # Support [0.1,0.2,...] or "0.1 0.2 ..."
            vec_str = vec_str.strip("[]")
            vec_str = vec_str.replace(",", " ")

            try:
                vec = np.array(
                    [float(x) for x in vec_str.split() if x.strip()],
                    dtype=np.float32,
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse vector at line {lineno}: {line[:200]}"
                ) from e

            if vec.size == 0:
                raise ValueError(f"Empty vector at line {lineno}")

            regexes.append(regex)
            vectors.append(vec)

    if not vectors:
        raise ValueError(f"No queries found in {path}")

    dim = int(vectors[0].size)
    for i, v in enumerate(vectors):
        if v.size != dim:
            raise ValueError(
                f"Inconsistent query vector dim at line {i + 1}: "
                f"expected {dim}, got {v.size}"
            )

    vectors = np.vstack(vectors).astype(np.float32)
    return regexes, vectors, dim


def read_groundtruth(path):
    """
    groundtruth.txt format:
        one query per line, ids separated by spaces

    Example:
        12 88 901 1024 ...
    """
    gt = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                gt.append([])
                continue

            try:
                ids = [int(x) for x in line.split() if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse groundtruth line {lineno}: {line[:200]}"
                ) from e

            gt.append(ids)

    return gt


def vec_to_pg(v):
    """
    Convert numpy vector to pgvector literal:
        [0.1,0.2,0.3]
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def metric_to_operator(metric):
    if metric == "l2":
        return "<->"
    elif metric == "ip":
        return "<#>"
    elif metric == "cosine":
        return "<=>"
    else:
        raise ValueError(f"Unknown metric: {metric}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", default="regexann_pg")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--table", default="items")

    parser.add_argument("--query", required=True)
    parser.add_argument("--groundtruth", required=True)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, required=True)
    parser.add_argument("--metric", choices=["l2", "ip", "cosine"], default="l2")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    regexes, qvecs, dim = read_query_txt(args.query)
    gt = read_groundtruth(args.groundtruth)

    n = min(len(regexes), len(qvecs), len(gt))
    if args.limit is not None:
        n = min(n, args.limit)

    print(f"[INFO] query file    = {args.query}")
    print(f"[INFO] groundtruth   = {args.groundtruth}")
    print(f"[INFO] regex queries = {len(regexes)}")
    print(f"[INFO] query vectors = {len(qvecs)}")
    print(f"[INFO] groundtruth   = {len(gt)}")
    print(f"[INFO] run queries   = {n}")
    print(f"[INFO] dim           = {dim}")
    print(f"[INFO] k             = {args.k}")
    print(f"[INFO] ef_search     = {args.ef_search}")
    print(f"[INFO] metric        = {args.metric}")

    op = metric_to_operator(args.metric)

    conn = psycopg2.connect(
        dbname=args.db,
        host=args.host,
        port=args.port,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Set HNSW ef_search
    cur.execute(f"SET hnsw.ef_search = {args.ef_search};")

    total_time = 0.0
    total_recall = 0.0

    rows = []

    for qi in tqdm(range(n), desc="Querying"):
        regex = regexes[qi]
        qv = vec_to_pg(qvecs[qi])

        truth = set(gt[qi][: args.k])

        sql = f"""
            SELECT id
            FROM {args.table}
            WHERE text ~ %s
            ORDER BY embedding {op} %s::vector
            LIMIT %s;
        """

        t0 = time.time()
        cur.execute(sql, (regex, qv, args.k))
        result = [int(r[0]) for r in cur.fetchall()]
        t1 = time.time()

        elapsed = t1 - t0
        total_time += elapsed

        hit = len(set(result) & truth)
        recall = hit / args.k if args.k > 0 else 0.0
        total_recall += recall

        rows.append(
            {
                "qid": qi,
                "regex": regex,
                "recall": recall,
                "time_ms": elapsed * 1000.0,
                "num_result": len(result),
                "result_ids": " ".join(map(str, result)),
            }
        )

    avg_recall = total_recall / n if n > 0 else 0.0
    avg_time_ms = total_time * 1000.0 / n if n > 0 else 0.0
    qps = n / total_time if total_time > 0 else 0.0
    recall_pct = avg_recall * 100.0

    print("[RESULT]")
    print(f"recall@{args.k} = {avg_recall:.6f}")
    print(f"recall_pct      = {recall_pct:.6f}")
    print(f"avg_time_ms     = {avg_time_ms:.3f}")
    print(f"qps             = {qps:.3f}")
    
    # CSV markers for bash grep
    print(f"RECALL_PCT,{recall_pct:.6f}")
    print(f"AVG_TIME_MS,{avg_time_ms:.6f}")
    print(f"QPS,{qps:.6f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "regex",
                "recall",
                "time_ms",
                "num_result",
                "result_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()