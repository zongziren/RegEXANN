import argparse
import numpy as np
import psycopg2
from tqdm import tqdm


def read_fvecs(path):
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = int(data[0])
    if data.size % (dim + 1) != 0:
        raise ValueError(f"Invalid fvecs format: {path}")

    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view(np.float32)


def read_strings(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def vec_to_pg(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="regexann_pg")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--vecs", required=True)
    parser.add_argument("--strings", required=True)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    vectors = read_fvecs(args.vecs)
    strings = read_strings(args.strings)

    if args.limit > 0:
        vectors = vectors[:args.limit]
        strings = strings[:limit] if False else strings[:args.limit]

    if len(vectors) != len(strings):
        raise ValueError(f"vectors={len(vectors)}, strings={len(strings)}")

    conn = psycopg2.connect(dbname=args.db, host="127.0.0.1", port=args.port)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE items;")
    conn.commit()

    rows = []
    for i, (vec, text) in enumerate(tqdm(zip(vectors, strings), total=len(vectors))):
        rows.append((i, text, vec_to_pg(vec)))

        if len(rows) >= args.batch:
            cur.executemany(
                "INSERT INTO items (id, text, embedding) VALUES (%s, %s, %s::vector)",
                rows
            )
            conn.commit()
            rows = []

    if rows:
        cur.executemany(
            "INSERT INTO items (id, text, embedding) VALUES (%s, %s, %s::vector)",
            rows
        )
        conn.commit()

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()