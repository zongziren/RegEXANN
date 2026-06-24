#!/usr/bin/env python3
import argparse
import time
import numpy as np
import psycopg2
from tqdm import tqdm


def read_fvecs(path):
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = int(data[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dim: {dim}")

    if data.size % (dim + 1) != 0:
        raise ValueError(
            f"Invalid fvecs format: {path}, "
            f"data.size={data.size}, dim={dim}"
        )

    data = data.reshape(-1, dim + 1)
    vectors = data[:, 1:].view(np.float32).copy()
    return vectors


def read_strings(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def vec_to_pg(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def metric_to_opclass(metric):
    if metric == "l2":
        return "vector_l2_ops"
    if metric == "ip":
        return "vector_ip_ops"
    if metric == "cosine":
        return "vector_cosine_ops"
    raise ValueError(f"Unknown metric: {metric}")


def get_table_size_mb(cur, table_name):
    cur.execute(
        "SELECT ROUND(pg_relation_size(%s) / 1024.0 / 1024.0, 3);",
        (table_name,),
    )
    return cur.fetchone()[0]


def get_table_total_size_mb(cur, table_name):
    cur.execute(
        "SELECT ROUND(pg_total_relation_size(%s) / 1024.0 / 1024.0, 3);",
        (table_name,),
    )
    return cur.fetchone()[0]


def get_index_size_mb(cur, table_name, index_name):
    """
    More robust than pg_relation_size(index_name) directly.
    It verifies that the index belongs to the target table.
    """
    cur.execute(
        """
        SELECT ROUND(pg_relation_size(indexrelid) / 1024.0 / 1024.0, 3)
        FROM pg_index
        WHERE indrelid = %s::regclass
          AND indexrelid = %s::regclass;
        """,
        (table_name, index_name),
    )
    row = cur.fetchone()
    if row is None:
        return "NA"
    return row[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", default="regexann_pg")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5433)

    parser.add_argument("--vecs", required=True)
    parser.add_argument("--strings", required=True)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--table", default="items")
    parser.add_argument("--recreate", action="store_true")

    parser.add_argument("--metric", choices=["l2", "ip", "cosine"], default="l2")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)

    args = parser.parse_args()

    vectors = read_fvecs(args.vecs)
    strings = read_strings(args.strings)

    if args.limit > 0:
        vectors = vectors[: args.limit]
        strings = strings[: args.limit]

    if len(vectors) != len(strings):
        raise ValueError(f"vectors={len(vectors)}, strings={len(strings)}")

    n, dim = vectors.shape
    opclass = metric_to_opclass(args.metric)

    print(f"[INFO] rows = {n}")
    print(f"[INFO] dim  = {dim}")
    print(f"[INFO] metric = {args.metric}")
    print(f"[INFO] HNSW m = {args.hnsw_m}")
    print(f"[INFO] HNSW ef_construction = {args.hnsw_ef_construction}")

    conn = psycopg2.connect(
        dbname=args.db,
        host=args.host,
        port=args.port,
    )
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    if args.recreate:
        print(f"[PG] Recreating table {args.table}")
        cur.execute(f"DROP TABLE IF EXISTS {args.table};")
        cur.execute(
            f"""
            CREATE TABLE {args.table} (
                id BIGINT PRIMARY KEY,
                text TEXT,
                embedding vector({dim})
            );
            """
        )
        conn.commit()
    else:
        print(f"[PG] Truncating table {args.table}")
        cur.execute(f"TRUNCATE TABLE {args.table};")
        conn.commit()

    print("[PG] Loading data")
    start_load = time.time()

    rows = []
    for i, (vec, text) in enumerate(tqdm(zip(vectors, strings), total=n)):
        rows.append((i, text, vec_to_pg(vec)))

        if len(rows) >= args.batch:
            cur.executemany(
                f"""
                INSERT INTO {args.table} (id, text, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                rows,
            )
            conn.commit()
            rows = []

    if rows:
        cur.executemany(
            f"""
            INSERT INTO {args.table} (id, text, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            rows,
        )
        conn.commit()

    load_time = time.time() - start_load
    print(f"[TIME] load_time_sec = {load_time:.3f}")

    index_name = f"{args.table}_embedding_hnsw_idx"

    print(f"[PG] Dropping old index {index_name}")
    cur.execute(f"DROP INDEX IF EXISTS {index_name};")
    conn.commit()

    print("[PG] Building HNSW index")
    start_build = time.time()

    cur.execute(
        f"""
        CREATE INDEX {index_name}
        ON {args.table}
        USING hnsw (embedding {opclass})
        WITH (
            m = {args.hnsw_m},
            ef_construction = {args.hnsw_ef_construction}
        );
        """
    )
    conn.commit()

    build_time = time.time() - start_build
    print(f"[TIME] hnsw_build_time_sec = {build_time:.3f}")

    print("[PG] Analyze table")
    cur.execute(f"ANALYZE {args.table};")
    conn.commit()

    print("[PG] Collecting size statistics")

    table_heap_mb = get_table_size_mb(cur, args.table)
    table_total_mb = get_table_total_size_mb(cur, args.table)
    hnsw_index_mb = get_index_size_mb(cur, args.table, index_name)

    print(f"TABLE_HEAP_MB,{table_heap_mb}")
    print(f"TABLE_TOTAL_MB,{table_total_mb}")
    print(f"HNSW_INDEX_MB,{hnsw_index_mb}")

    cur.close()
    conn.close()

    print("[DONE]")


if __name__ == "__main__":
    main()