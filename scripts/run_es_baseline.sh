#!/usr/bin/env bash
set -euo pipefail

HOST=http://localhost:9200
K=10

DATASETS=(
  arxiv
  audio
  dbpedia
  gist
  laion
  msong
  sift
  words
)

CANDIDATES=(100 300 500 1000 2000 5000 10000)

for ds in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: ${ds}"
    echo "========================================"

    VEC="dataset/${ds}/vectors.fvecs"
    STR="dataset/${ds}/strings.txt"
    QRY="dataset/${ds}/query.txt"
    GT="dataset/${ds}/groundtruth.txt"

    OUTDIR="results/${ds}/es"
    mkdir -p "${OUTDIR}"

    INDEX="regexann_${ds}"

    echo "[Check files]"
    ls -lh "${VEC}" "${STR}" "${QRY}" "${GT}"

    echo "[1/2] Build Elasticsearch index: ${INDEX}"

    python3 scripts/es_index.py \
        --host "${HOST}" \
        --index "${INDEX}" \
        --vecs "${VEC}" \
        --strings "${STR}" \
        --batch 1000 \
        | tee "${OUTDIR}/build.log"

    echo "method,param,param_value,recall_pct,avg_time_ms,qps" > "${OUTDIR}/summary.csv"

    echo "[2/2] Run Elasticsearch baseline"

    for nc in "${CANDIDATES[@]}"; do
        echo "----------------------------------------"
        echo "Dataset=${ds}, num_candidates=${nc}"
        echo "----------------------------------------"

        LOG="${OUTDIR}/query_nc${nc}.log"
        RAW="${OUTDIR}/raw_nc${nc}.csv"

        python3 scripts/es_query.py \
            --host "${HOST}" \
            --index "${INDEX}" \
            --vecs "${VEC}" \
            --query "${QRY}" \
            --groundtruth "${GT}" \
            --out "${RAW}" \
            --k "${K}" \
            --num_candidates "${nc}" \
            | tee "${LOG}"

        recall=$(grep "RECALL_PCT" "${LOG}" | tail -1 | cut -d',' -f2)
        avg=$(grep "AVG_TIME_MS" "${LOG}" | tail -1 | cut -d',' -f2)
        qps=$(grep "QPS" "${LOG}" | tail -1 | cut -d',' -f2)

        echo "elasticsearch,num_candidates,${nc},${recall},${avg},${qps}" >> "${OUTDIR}/summary.csv"
    done
done

