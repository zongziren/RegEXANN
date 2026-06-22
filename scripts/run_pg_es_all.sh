#!/usr/bin/env bash
set -euo pipefail

PYTHON=/usr/bin/python3

PG_DB="${PG_DB:-regexann_pg}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5433}"

K="${K:-10}"
PG_BATCH="${PG_BATCH:-1000}"

ES_HOST="${ES_HOST:-http://localhost:9200}"
ES_NUM_CANDIDATES="${ES_NUM_CANDIDATES:-10000}"

BUILD_ES="${BUILD_ES:-0}"
BUILD_PG="${BUILD_PG:-1}"

mkdir -p results/baselines

get_dim() {
    local fvec="$1"
    "${PYTHON}" - "$fvec" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
data = np.fromfile(path, dtype=np.int32, count=1)
print(int(data[0]))
PY
}

now_sec() {
    date +%s.%N
}

elapsed_sec() {
    local start="$1"
    local end="$2"
    "${PYTHON}" - <<PY
start = float("${start}")
end = float("${end}")
print(f"{end - start:.3f}")
PY
}

get_pg_mem_mb() {
    pgrep -u "$USER" -f "postgres.*${PG_PORT}" | head -1 | xargs -r ps -o rss= -p | awk '{printf "%.2f\n", $1/1024}'
}

get_es_mem_mb() {
    pgrep -u "$USER" -f "org.elasticsearch.bootstrap.Elasticsearch" | head -1 | xargs -r ps -o rss= -p | awk '{printf "%.2f\n", $1/1024}'
}

get_pg_size_mb() {
    local table="$1"
    psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -Atc \
        "SELECT ROUND(pg_total_relation_size('${table}') / 1024.0 / 1024.0, 2);"
}

get_es_size_mb() {
    local index="$1"
    curl -s "${ES_HOST}/${index}/_stats/store" | "${PYTHON}" - <<'PY'
import sys, json
try:
    obj = json.load(sys.stdin)
    size = obj["_all"]["total"]["store"]["size_in_bytes"] / 1024 / 1024
    print(f"{size:.2f}")
except Exception:
    print("NA")
PY
}

run_one() {
    local name="$1"
    local strings="$2"
    local vecs="$3"
    local query="$4"
    local gt="$5"

    echo "════════════════════════════════════════════════════════════"
    echo "Dataset: ${name}"
    echo "strings     = ${strings}"
    echo "vectors     = ${vecs}"
    echo "query       = ${query}"
    echo "groundtruth = ${gt}"
    echo "════════════════════════════════════════════════════════════"

    if [[ ! -f "${strings}" ]]; then
        echo "[SKIP] missing strings file: ${strings}"
        echo ""
        return 0
    fi

    if [[ ! -f "${vecs}" ]]; then
        echo "[SKIP] missing vector file: ${vecs}"
        echo ""
        return 0
    fi

    if [[ ! -f "${query}" ]]; then
        echo "[SKIP] missing query file: ${query}"
        echo ""
        return 0
    fi

    if [[ ! -f "${gt}" ]]; then
        echo "[SKIP] missing groundtruth file: ${gt}"
        echo ""
        return 0
    fi

    local dim
    dim="$(get_dim "${vecs}")"

    echo "[INFO] dim = ${dim}"

    mkdir -p "results/${name}/pgvector"
    mkdir -p "results/${name}/es"

    # ─────────────────────────────────────────────────────────────
    # PostgreSQL + pgvector exact pre-filter baseline
    # ─────────────────────────────────────────────────────────────

    if [[ "${BUILD_PG}" == "1" ]]; then
        echo "[PG] Recreating table items with vector(${dim})"

        psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -c "CREATE EXTENSION IF NOT EXISTS vector;"
        psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -c "DROP TABLE IF EXISTS items;"
        psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -c "
CREATE TABLE items (
    id integer PRIMARY KEY,
    text text,
    embedding vector(${dim})
);
"

        echo "[PG] Loading dataset ${name}"

        local pg_start
        local pg_end
        local pg_time
        pg_start="$(now_sec)"

        "${PYTHON}" scripts/pg_load.py \
            --db "${PG_DB}" \
            --port "${PG_PORT}" \
            --vecs "${vecs}" \
            --strings "${strings}" \
            --batch "${PG_BATCH}" \
            | tee "results/${name}/pgvector/load.log"

        pg_end="$(now_sec)"
        pg_time="$(elapsed_sec "${pg_start}" "${pg_end}")"

        local pg_size
        local pg_mem
        pg_size="$(get_pg_size_mb items)"
        pg_mem="$(get_pg_mem_mb || true)"

        {
            echo "PG_BUILD_TIME_SEC,${pg_time}"
            echo "PG_INDEX_SIZE_MB,${pg_size}"
            echo "PG_MEM_RSS_MB,${pg_mem:-NA}"
        } | tee -a "results/${name}/pgvector/load.log"
    else
        echo "[PG] Skip build/load. Use existing table items."
    fi

    echo "[PG] Querying dataset ${name}"

    "${PYTHON}" scripts/pg_query.py \
        --db "${PG_DB}" \
        --host "${PG_HOST}" \
        --port "${PG_PORT}" \
        --vecs "${vecs}" \
        --query "${query}" \
        --groundtruth "${gt}" \
        --out "results/${name}/pgvector/raw.csv" \
        --k "${K}" \
        | tee "results/${name}/pgvector/query.log"

    local pg_query_mem
    pg_query_mem="$(get_pg_mem_mb || true)"
    echo "PG_QUERY_MEM_RSS_MB,${pg_query_mem:-NA}" | tee -a "results/${name}/pgvector/query.log"

    # ─────────────────────────────────────────────────────────────
    # Elasticsearch filtered-kNN baseline
    # ─────────────────────────────────────────────────────────────

    local es_index="regexann_${name}"

    if [[ "${BUILD_ES}" == "1" ]]; then
        echo "[ES] Rebuilding index ${es_index}"

        local es_start
        local es_end
        local es_time
        es_start="$(now_sec)"

        "${PYTHON}" scripts/es_index.py \
            --host "${ES_HOST}" \
            --index "${es_index}" \
            --vecs "${vecs}" \
            --strings "${strings}" \
            | tee "results/${name}/es/index.log"

        es_end="$(now_sec)"
        es_time="$(elapsed_sec "${es_start}" "${es_end}")"

        local es_size
        local es_mem
        es_size="$(get_es_size_mb "${es_index}")"
        es_mem="$(get_es_mem_mb || true)"

        {
            echo "ES_BUILD_TIME_SEC,${es_time}"
            echo "ES_INDEX_SIZE_MB,${es_size}"
            echo "ES_MEM_RSS_MB,${es_mem:-NA}"
        } | tee -a "results/${name}/es/index.log"
    else
        echo "[ES] Skip index build. Use existing index: ${es_index}"
    fi

    echo "[ES] Querying dataset ${name}"

    "${PYTHON}" scripts/es_query.py \
        --host "${ES_HOST}" \
        --index "${es_index}" \
        --vecs "${vecs}" \
        --query "${query}" \
        --groundtruth "${gt}" \
        --out "results/${name}/es/raw.csv" \
        --k "${K}" \
        --num_candidates "${ES_NUM_CANDIDATES}" \
        | tee "results/${name}/es/query.log"

    local es_query_mem
    es_query_mem="$(get_es_mem_mb || true)"
    echo "ES_QUERY_MEM_RSS_MB,${es_query_mem:-NA}" | tee -a "results/${name}/es/query.log"

    echo "[DONE] ${name}"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo " Running pgvector + Elasticsearch baselines"
echo " K                 = ${K}"
echo " PG_DB             = ${PG_DB}"
echo " PG_HOST           = ${PG_HOST}"
echo " PG_PORT           = ${PG_PORT}"
echo " BUILD_PG          = ${BUILD_PG}"
echo " ES_HOST           = ${ES_HOST}"
echo " ES_NUM_CANDIDATES = ${ES_NUM_CANDIDATES}"
echo " BUILD_ES          = ${BUILD_ES}"
echo "════════════════════════════════════════════════════════════"
echo ""

run_one arxiv \
    dataset/arxiv/strings.txt \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/query.txt \
    dataset/arxiv/groundtruth.txt

run_one words \
    dataset/words/strings.txt \
    dataset/words/vectors.fvecs \
    dataset/words/query.txt \
    dataset/words/groundtruth.txt

run_one dbpedia \
    dataset/dbpedia/strings.txt \
    dataset/dbpedia/vectors.fvecs \
    dataset/dbpedia/query.txt \
    dataset/dbpedia/groundtruth.txt

run_one sift \
    dataset/sift/sift_titles_clean.txt \
    dataset/sift/sift_vectors.fvecs \
    dataset/sift/query.txt \
    dataset/sift/groundtruth.txt

run_one gist \
    dataset/gist/strings.txt \
    dataset/gist/vectors.fvecs \
    dataset/gist/query.txt \
    dataset/gist/groundtruth.txt

run_one laion \
    dataset/laion/strings.txt \
    dataset/laion/vectors.fvecs \
    dataset/laion/query.txt \
    dataset/laion/groundtruth.txt

run_one msong \
    dataset/msong/strings.txt \
    dataset/msong/vectors.fvecs \
    dataset/msong/query.txt \
    dataset/msong/groundtruth.txt

run_one audio \
    dataset/audio/strings.txt \
    dataset/audio/vectors.fvecs \
    dataset/audio/query.txt \
    dataset/audio/groundtruth.txt

echo "════════════════════════════════════════════════════════════"
echo " All done."
echo " Results:"
echo "   results/<dataset>/pgvector/load.log"
echo "   results/<dataset>/pgvector/query.log"
echo "   results/<dataset>/es/index.log"
echo "   results/<dataset>/es/query.log"
echo "════════════════════════════════════════════════════════════"
