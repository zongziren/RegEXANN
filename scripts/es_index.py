import argparse
import time
import numpy as np
from elasticsearch import Elasticsearch, helpers


def read_fvecs(path):
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = int(data[0])
    if data.size % (dim + 1) != 0:
        raise ValueError(
            f"Invalid fvecs format: {path}, dim={dim}, size={data.size}"
        )

    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view(np.float32)


def read_strings(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def create_index(es, index_name, dim):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1"
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "integer"},
                "text": {
                    "type": "keyword",
                    "ignore_above": 32766
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "l2_norm"
                }
            }
        }
    }

    es.indices.create(index=index_name, body=body)


def bulk_index(es, index_name, vectors, strings, batch_size):
    if len(vectors) != len(strings):
        raise ValueError(f"vectors={len(vectors)}, strings={len(strings)}")

    def actions():
        for i, (vec, text) in enumerate(zip(vectors, strings)):
            yield {
                "_index": index_name,
                "_id": i,
                "_source": {
                    "doc_id": i,
                    "text": text,
                    "vector": vec.tolist()
                }
            }

    helpers.bulk(
        es,
        actions(),
        chunk_size=batch_size,
        request_timeout=300,
        max_retries=5,
        initial_backoff=2,
        max_backoff=30
    )

    es.indices.put_settings(
        index=index_name,
        body={"index": {"refresh_interval": "1s"}}
    )
    es.indices.refresh(index=index_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--vecs", required=True)
    parser.add_argument("--strings", required=True)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()

    es = Elasticsearch(args.host, request_timeout=300)

    print(f"Loading vectors: {args.vecs}", flush=True)
    vectors = read_fvecs(args.vecs)

    print(f"Loading strings: {args.strings}", flush=True)
    strings = read_strings(args.strings)

    print(f"Vector shape: {vectors.shape}", flush=True)
    print(f"String count: {len(strings)}", flush=True)

    print(f"Creating index: {args.index}", flush=True)
    create_index(es, args.index, vectors.shape[1])

    print("Bulk indexing...", flush=True)
    start = time.time()
    bulk_index(es, args.index, vectors, strings, args.batch)
    build_time = time.time() - start

    stats = es.indices.stats(index=args.index, metric="store")
    store_size = stats["indices"][args.index]["total"]["store"]["size_in_bytes"]

    print(f"BUILD_TIME_SEC,{build_time:.3f}", flush=True)
    print(f"STORE_SIZE_MB,{store_size / 1024 / 1024:.2f}", flush=True)


if __name__ == "__main__":
    main()