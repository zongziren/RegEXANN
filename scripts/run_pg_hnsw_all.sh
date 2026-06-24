#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

DB=regexann_pg
HOST=127.0.0.1
PORT=5433
TABLE=items

K=10
BATCH=1000
LIMIT=-1

DATASETS=(
  arxiv
  sift
  gist
  laion
  audio
  msong
  dbpedia
  words
)

# arxiv 已经 load/build 过了，这里只 query，不重新 load
SKIP_LOAD_DATASETS=(
  arxiv
)

EF_LIST=(10 20 40 80 100 200 400 800 1600)

ROOT_OUTDIR="results/pg_hnsw_all"
mkdir -p "${ROOT_OUTDIR}"

ALL_SUMMARY="${ROOT_OUTDIR}/summary_all.csv"
BUILD_SUMMARY="${ROOT_OUTDIR}/build_summary.csv"

# append 模式：只有文件不存在时才写 header
if [[ ! -f "${ALL_SUMMARY}" ]]; then
    echo "dataset,method,param,param_value,recall_pct,avg_time_ms,qps" > "${ALL_SUMMARY}"
fi

if [[ ! -f "${BUILD_SUMMARY}" ]]; then
    echo "dataset,build_time_sec,max_rss_kb,max_rss_mb,table_heap_mb,table_total_mb,hnsw_index_mb,total_table_index_mb" > "${BUILD_SUMMARY}"
fi

should_skip_load() {
    local x="$1"
    for y in "${SKIP_LOAD_DATASETS[@]}"; do
        if [[ "${x}" == "${y}" ]]; then
            return 0
        fi
    done
    return 1
}

for DATASET in "${DATASETS[@]}"; do
    VEC="dataset/${DATASET}/vectors.fvecs"
    STR="dataset/${DATASET}/strings.txt"
    QUERY="dataset/${DATASET}/query.txt"
    GT="dataset/${DATASET}/groundtruth.txt"

    OUTDIR="${ROOT_OUTDIR}/${DATASET}"
    mkdir -p "${OUTDIR}"

    SUMMARY="${OUTDIR}/summary.csv"

    # append 模式：dataset 自己的 summary 也不覆盖
    if [[ ! -f "${SUMMARY}" ]]; then
        echo "method,param,param_value,recall_pct,avg_time_ms,qps" > "${SUMMARY}"
    fi

    echo
    echo "════════════════════════════════════════════════════════════"
    echo "Dataset: ${DATASET}"
    echo "vectors     = ${VEC}"
    echo "strings     = ${STR}"
    echo "query       = ${QUERY}"
    echo "groundtruth = ${GT}"
    echo "outdir      = ${OUTDIR}"
    echo "════════════════════════════════════════════════════════════"

    if [[ ! -f "${VEC}" ]]; then
        echo "[SKIP] Missing vectors: ${VEC}"
        continue
    fi

    if [[ ! -f "${STR}" ]]; then
        echo "[SKIP] Missing strings: ${STR}"
        continue
    fi

    if [[ ! -f "${QUERY}" ]]; then
        echo "[SKIP] Missing query: ${QUERY}"
        continue
    fi

    if [[ ! -f "${GT}" ]]; then
        echo "[SKIP] Missing groundtruth: ${GT}"
        continue
    fi

    if should_skip_load "${DATASET}"; then
        echo
        echo "────────────────────────────────────────────────────────────"
        echo "[1/2] Skip loading/building HNSW index: ${DATASET}"
        echo "────────────────────────────────────────────────────────────"
        echo "[SKIP LOAD] ${DATASET} is assumed to be already loaded in PostgreSQL table: ${TABLE}"
        echo "[SKIP LOAD] Make sure current table '${TABLE}' contains dataset '${DATASET}'"
    else
        echo
        echo "────────────────────────────────────────────────────────────"
        echo "[1/2] Loading data and building HNSW index: ${DATASET}"
        echo "────────────────────────────────────────────────────────────"

        BUILD_LOG="${OUTDIR}/build.log"
        BUILD_TIME_LOG="${OUTDIR}/build_time.log"

        /usr/bin/time -v -o "${BUILD_TIME_LOG}" \
        python3 scripts/pg_load_hnsw.py \
            --db "${DB}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --table "${TABLE}" \
            --vecs "${VEC}" \
            --strings "${STR}" \
            --batch "${BATCH}" \
            --limit "${LIMIT}" \
            --recreate \
            --metric l2 \
            --hnsw-m 16 \
            --hnsw-ef-construction 200 \
            2>&1 | tee "${BUILD_LOG}"

        BUILD_TIME=$(grep "^\\[TIME\\] hnsw_build_time_sec" "${BUILD_LOG}" | tail -1 | awk -F'= ' '{print $2}' || echo "NA")
        MAX_RSS_KB=$(grep "Maximum resident set size" "${BUILD_TIME_LOG}" | awk '{print $6}' || echo "NA")

        if [[ "${MAX_RSS_KB}" != "NA" && -n "${MAX_RSS_KB}" ]]; then
            MAX_RSS_MB=$(awk "BEGIN {printf \"%.3f\", ${MAX_RSS_KB}/1024}")
        else
            MAX_RSS_MB="NA"
        fi

        TABLE_HEAP_MB=$(grep "^TABLE_HEAP_MB," "${BUILD_LOG}" | tail -1 | cut -d',' -f2 || echo "NA")
        TABLE_TOTAL_MB=$(grep "^TABLE_TOTAL_MB," "${BUILD_LOG}" | tail -1 | cut -d',' -f2 || echo "NA")
        HNSW_INDEX_MB=$(grep "^HNSW_INDEX_MB," "${BUILD_LOG}" | tail -1 | cut -d',' -f2 || echo "NA")
        TOTAL_TABLE_INDEX_MB="${TABLE_TOTAL_MB}"

        echo "${DATASET},${BUILD_TIME},${MAX_RSS_KB},${MAX_RSS_MB},${TABLE_HEAP_MB},${TABLE_TOTAL_MB},${HNSW_INDEX_MB},${TOTAL_TABLE_INDEX_MB}" >> "${BUILD_SUMMARY}"

        echo
        echo "Build stats:"
        echo "  build_time_sec       = ${BUILD_TIME}"
        echo "  max_rss_mb           = ${MAX_RSS_MB}"
        echo "  table_heap_mb        = ${TABLE_HEAP_MB}"
        echo "  table_total_mb       = ${TABLE_TOTAL_MB}"
        echo "  hnsw_index_mb        = ${HNSW_INDEX_MB}"
    fi

    echo
    echo "────────────────────────────────────────────────────────────"
    echo "[2/2] Sweeping ef_search: ${DATASET}"
    echo "────────────────────────────────────────────────────────────"

    for EF in "${EF_LIST[@]}"; do
        echo
        echo "→ dataset=${DATASET}, ef_search=${EF}"

        LOG="${OUTDIR}/ef${EF}.log"
        OUT="${OUTDIR}/ef${EF}_results.csv"

        python3 scripts/pg_query_hnsw.py \
            --db "${DB}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --table "${TABLE}" \
            --query "${QUERY}" \
            --groundtruth "${GT}" \
            --out "${OUT}" \
            --k "${K}" \
            --ef-search "${EF}" \
            --metric l2 \
            2>&1 | tee "${LOG}"

        RECALL=$(grep "^RECALL_PCT," "${LOG}" | tail -1 | cut -d',' -f2 || echo "NA")
        AVGMS=$(grep "^AVG_TIME_MS," "${LOG}" | tail -1 | cut -d',' -f2 || echo "NA")
        QPS=$(grep "^QPS," "${LOG}" | tail -1 | cut -d',' -f2 || echo "NA")

        echo "pgvector_hnsw,ef_search,${EF},${RECALL},${AVGMS},${QPS}" >> "${SUMMARY}"
        echo "${DATASET},pgvector_hnsw,ef_search,${EF},${RECALL},${AVGMS},${QPS}" >> "${ALL_SUMMARY}"
    done

    echo
    echo "[DONE] ${DATASET}"
    echo "summary       = ${SUMMARY}"
    echo "all summary   = ${ALL_SUMMARY}"
    echo "build summary = ${BUILD_SUMMARY}"
done

echo
echo "════════════════════════════════════════════════════════════"
echo "[ALL DONE]"
echo "Query summary = ${ALL_SUMMARY}"
echo "Build summary = ${BUILD_SUMMARY}"
echo "════════════════════════════════════════════════════════════"