#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/usr/bin/python3}"

PG_DB="${PG_DB:-regexann_pg}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5433}"

K="${K:-10}"
PG_BATCH="${PG_BATCH:-1000}"
BUILD_PG="${BUILD_PG:-1}"

get_dim() {
    local fvec="$1"
    "${PYTHON}" - "$fvec" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
with open(path, "rb") as f:
    dim = np.fromfile(f, dtype=np.int32, count=1)[0]
print(int(dim))
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
    local pid
    pid="$(pgrep -u "$USER" -f "postgres.*${PG_PORT}" | head -1 || true)"
    if [[ -z "${pid}" ]]; then
        echo "NA"
        return 0
    fi
    ps -o rss= -p "${pid}" | awk '{printf "%.2f\n", $1/1024}'
}

get_pg_size_mb() {
    local table="$1"
    psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -Atc \
        "SELECT ROUND(pg_total_relation_size('${table}') / 1024.0 / 1024.0, 2);"
}

check_pg_alive() {
    if ! psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" -Atc "SELECT 1;" >/dev/null; then
        echo "[ERROR] PostgreSQL is not reachable."
        echo "        PG_DB   = ${PG_DB}"
        echo "        PG_HOST = ${PG_HOST}"
        echo "        PG_PORT = ${PG_PORT}"
        exit 1
    fi
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

    if [[ "${BUILD_PG}" == "1" ]]; then
        echo "[PG] Recreating table items with vector(${dim})"

        psql -h "${PG_HOST}" -p "${PG_PORT}" -d "${PG_DB}" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS items;
CREATE TABLE items (
    id integer PRIMARY KEY,
    text text,
    embedding vector(${dim})
);
SQL

        echo "[PG] Loading dataset ${name}"

        local pg_start pg_end pg_time
        pg_start="$(now_sec)"

        "${PYTHON}" scripts/pg_load.py \
            --db "${PG_DB}" \
            --port "${PG_PORT}" \
            --vecs "${vecs}" \
            --strings "${strings}" \
            --batch "${PG_BATCH}" \
            2>&1 | tee "results/${name}/pgvector/load.log"

        pg_end="$(now_sec)"
        pg_time="$(elapsed_sec "${pg_start}" "${pg_end}")"

        local pg_size pg_mem
        pg_size="$(get_pg_size_mb items || true)"
        pg_mem="$(get_pg_mem_mb || true)"

        {
            echo "PG_BUILD_TIME_SEC,${pg_time}"
            echo "PG_INDEX_SIZE_MB,${pg_size:-NA}"
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
        2>&1 | tee "results/${name}/pgvector/query.log"

    local pg_query_mem
    pg_query_mem="$(get_pg_mem_mb || true)"
    echo "PG_QUERY_MEM_RSS_MB,${pg_query_mem:-NA}" | tee -a "results/${name}/pgvector/query.log"

    echo "[DONE] ${name}"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo " Running pgvector baseline only"
echo " K        = ${K}"
echo " PG_DB    = ${PG_DB}"
echo " PG_HOST  = ${PG_HOST}"
echo " PG_PORT  = ${PG_PORT}"
echo " PG_BATCH = ${PG_BATCH}"
echo " BUILD_PG = ${BUILD_PG}"
echo "════════════════════════════════════════════════════════════"
echo ""

check_pg_alive

#run_one arxiv \
#    dataset/arxiv/strings.txt \
#    dataset/arxiv/vectors.fvecs \
#    dataset/arxiv/query.txt \
#    dataset/arxiv/groundtruth.txt

#run_one words \
#    dataset/words/strings.txt \
#    dataset/words/vectors.fvecs \
#    dataset/words/query.txt \
#    dataset/words/groundtruth.txt

#run_one dbpedia \
#    dataset/dbpedia/strings.txt \
#    dataset/dbpedia/vectors.fvecs \
#    dataset/dbpedia/query.txt \
#    dataset/dbpedia/groundtruth.txt

run_one sift \
    dataset/sift/strings.txt \
    dataset/sift/vectors.fvecs \
    dataset/sift/query.txt \
    dataset/sift/groundtruth.txt

#run_one gist \
#    dataset/gist/strings.txt \
#    dataset/gist/vectors.fvecs \
#    dataset/gist/query.txt \
#    dataset/gist/groundtruth.txt

#run_one laion \
#    dataset/laion/strings.txt \
#    dataset/laion/vectors.fvecs \
#    dataset/laion/query.txt \
#    dataset/laion/groundtruth.txt

#run_one msong \
#    dataset/msong/strings.txt \
#    dataset/msong/vectors.fvecs \
#    dataset/msong/query.txt \
#    dataset/msong/groundtruth.txt

#run_one audio \
#    dataset/audio/strings.txt \
#    dataset/audio/vectors.fvecs \
#    dataset/audio/query.txt \
#    dataset/audio/groundtruth.txt

echo "════════════════════════════════════════════════════════════"
echo " All done."
echo " Results:"
echo "   results/<dataset>/pgvector/load.log"
echo "   results/<dataset>/pgvector/query.log"
echo "   results/<dataset>/pgvector/raw.csv"
echo "════════════════════════════════════════════════════════════"
