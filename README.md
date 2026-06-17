# RegExANN — Compilation and Running Guide

## 1. Environment Requirements

| Tool          | Minimum Version | Purpose                   |
| ------------- | --------------- | ------------------------- |
| g++ / clang++ | C++17 support   | Compilation               |
| cmake         | 3.10+           | Build system              |
| python3       | 3.7+            | Auxiliary tools, optional |

Ubuntu / Debian one-command installation:

```bash
sudo apt-get install -y build-essential cmake
```

---

## 2. Compilation

```bash
cd RegExANN/exp

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build outputs:
#   build/regann        Main program for search and index construction
#   build/eval_recall   Standalone Recall@K evaluation tool
```

---

## 3. Data Format

### 3.1 Vector File `.fvecs`

The standard `fvecs` binary format is used. Each vector is stored as:

```text
[int32: dim][float32 × dim]
```

The `bvecs` format (`uint8` vectors) is also supported via `fmt=bvecs`.

### 3.2 String File `strings.txt`

One string per line, corresponding 1-to-1 with the vector file.

```text
deep learning for image retrieval
approximate nearest neighbor search
graph neural network embedding
...
```

### 3.3 Query File `queries.txt`

One query per line: a regular expression, then a space, then the query vector components.

```text
deep.*learning 0.123 -0.456 0.789 ...
neural|network 0.001 0.234 -0.567 ...
(ann|knn).*gpu 0.111 0.222 0.333 ...
```

---

## 4. Main Program Usage

```bash
./build/regann <vectors.fvecs> <strings.txt> <queries.txt> <K> \
               <clusters> <output.txt> <max_iter> <algorithm> [options...]
```

| Argument        | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `vectors.fvecs` | Dataset vector file                                           |
| `strings.txt`   | Dataset string file, corresponding to the vectors             |
| `queries.txt`   | Query file                                                    |
| `K`             | Number of nearest neighbors to return                         |
| `clusters`      | Number of k-means clusters. Recommended range: `50` to `500`  |
| `output.txt`    | Output file. Each line stores the result IDs for one query    |
| `max_iter`      | Maximum number of k-means iterations. Recommended value: `30` |
| `algorithm`     | Algorithm option. See the table below                         |

### 4.1 Algorithm Options

| Algorithm     | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `ann`         | **RegExANN**, the main method: k-means + trigram index + PQ        |
| `hier`        | Two-level hierarchical RegExANN                                    |
| `groundtruth` | Exact full scan, used to generate ground truth                     |
| `baseline`    | Same as `groundtruth`                                              |
| `prefilter`   | Pre-filtering baseline: regex filter first, then kNN               |
| `postfilter`  | Post-filtering baseline: kNN first, then regex filter              |

### 4.2 Optional Parameters

All options use `key=value` format.

| Option                | Applicable Algorithms | Default          | Description                                                                                                                                                   |
| --------------------- | --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pq_m=N`              | `ann`, `hier`         | `8`              | Number of PQ subspaces. Must divide the vector dimension.                                                                                                     |
| `pq_ksub=N`           | `ann`, `hier`         | `256`            | Codebook size for each PQ subspace.                                                                                                                           |
| `ef=N`                | `ann`                 | `K`              | **Candidate pool size (≥ K).** The algorithm collects `ef` regex-matching candidates before stopping, then returns the top-K by exact distance. Larger `ef` → higher recall, slower query. `ef=K` is the original behaviour. |
| `k0=N`                | `hier`                | `sqrt(clusters)` | Number of coarse clusters.                                                                                                                                    |
| `nprobe=N`            | `hier`                | `k0/2`           | Number of coarse clusters to probe during search.                                                                                                             |
| `oversample=N`        | `postfilter`          | `10`             | Initial oversampling factor. The first candidate window is `K × N`.                                                                                           |
| `max_expansion=N`     | `postfilter`          | `0`              | **Maximum expansion cap.** If fewer than K regex matches are found in the initial window, the window doubles repeatedly until K matches are found or `K × N` total candidates have been scanned. `0` = no cap (scan full dataset if needed). |
| `sample_ratio=F`      | `prefilter`           | `1.0`            | **Fraction of the regex-matching set to search (0 < F ≤ 1.0).** `1.0` = search all matches (100% recall, original behaviour). Values below 1.0 randomly subsample the match set, trading recall for speed. |
| `gt=<file>`           | All                   | None             | Ground truth file. If provided, Recall@K is computed and printed automatically.                                                                               |
| `save=<prefix>`       | `ann`                 | None             | Save the index as `<prefix>.{kmidx,gramidx,pqidx}`.                                                                                                          |
| `load=<prefix>`       | `ann`                 | None             | Skip index construction and load a saved index.                                                                                                               |
| `fmt=fvecs\|bvecs`    | All                   | `fvecs`          | Vector file format.                                                                                                                                           |
| `graph_M=N`           | `graph`               | `16`             | Graph max degree.                                                                                                                                             |
| `graph_ef_build=N`    | `graph`               | `64`             | Graph build beam width.                                                                                                                                       |
| `graph_ef_search=N`   | `graph`               | `64`             | Graph search beam width.                                                                                                                                       |

---

## 5. Recall / Speed Trade-off Parameters

Three new parameters give fine-grained control over the recall–speed curve for each algorithm:

### 5.1 `ann` — `ef` (candidate pool size)

`ef` is analogous to `efSearch` in HNSW. The search collects `ef` regex-matching candidates (instead of stopping at K), re-ranks them by exact distance, and returns the top-K.

```bash
# Original behaviour (ef = K = 10)
./build/regann ... ann gt=gt.txt

# Higher recall: collect 50 candidates, return top-10
./build/regann ... ann ef=50 gt=gt.txt

# Maximum recall sweep
for EF in 10 20 50 100 200; do
    ./build/regann ... ann ef=${EF} gt=gt.txt
done
```

Typical guidance: start at `ef=2K`, increase until recall meets your target.

### 5.2 `postfilter` — adaptive expansion

The classic fixed-oversample approach fails silently when regex selectivity is high (few matches in the dataset): `K × oversample` global neighbors may contain zero matches. The adaptive expansion variant doubles the search window until K matches are found.

```bash
# Fixed window, no expansion (original behaviour)
./build/regann ... postfilter oversample=10 max_expansion=10 gt=gt.txt

# Adaptive: expand until K matches found or full scan
./build/regann ... postfilter oversample=10 max_expansion=0 gt=gt.txt

# Adaptive with a cost cap: expand up to K×200 candidates
./build/regann ... postfilter oversample=10 max_expansion=200 gt=gt.txt
```

### 5.3 `prefilter` — `sample_ratio`

Pre-filtering applies regex first (perfect recall on the match set) then runs exact kNN. With `sample_ratio < 1.0`, only a random fraction of the matching vectors are searched, enabling a recall–speed trade-off.

```bash
# Original: search 100% of matching vectors (recall = 100%)
./build/regann ... prefilter sample_ratio=1.0 gt=gt.txt

# Search 20% of matches (faster, lower recall)
./build/regann ... prefilter sample_ratio=0.2 gt=gt.txt

# Sweep
for R in 1.0 0.5 0.2 0.1; do
    ./build/regann ... prefilter sample_ratio=${R} gt=gt.txt
done
```

---

## 6. Typical Workflow

### Step 1: Generate Ground Truth

```bash
./build/regann \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/strings.txt \
    dataset/arxiv/queries.txt \
    10 100 \
    results/arxiv/gt.txt \
    30 groundtruth
```

### Step 2: Run RegExANN and Save the Index

```bash
./build/regann \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/strings.txt \
    dataset/arxiv/queries.txt \
    10 100 \
    results/arxiv/ann.txt \
    30 ann \
    pq_m=8 ef=50 save=results/arxiv/idx/arxiv gt=results/arxiv/gt.txt
```

Example output:

```text
[INFO] Dataset: 132687 vectors (dim=768), 132687 strings.
[INFO] Building index (k=100, pq_m=8, pq_ksub=256) …
[INFO] Trigram index: 3821 trigrams.
[INFO] Index built in 4231 ms.
[INFO] Memory: 312.5 MB
[INFO] ef=50
[INFO] Queries          : 1000
[INFO] Avg total time   : 2.14 ms
[INFO] Avg set-op time  : 0.08 ms
[INFO] Avg cluster time : 2.05 ms
[INFO] QPS              : 467.3
[EVAL] Recall@10 = 94.7 %
```

### Step 3: Load the Index on Subsequent Runs

```bash
./build/regann \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/strings.txt \
    dataset/arxiv/queries.txt \
    10 100 \
    results/arxiv/ann2.txt \
    30 ann \
    load=results/arxiv/idx/arxiv ef=50 gt=results/arxiv/gt.txt
```

### Step 4: Run Baseline Methods

```bash
# Pre-filtering — original (100% recall)
./build/regann ... 30 prefilter gt=results/arxiv/gt.txt

# Pre-filtering — 30% sample (faster, lower recall)
./build/regann ... 30 prefilter sample_ratio=0.3 gt=results/arxiv/gt.txt

# Post-filtering — adaptive expansion (no cap)
./build/regann ... 30 postfilter oversample=10 max_expansion=0 gt=results/arxiv/gt.txt

# Post-filtering — fixed window (original, oversample=20)
./build/regann ... 30 postfilter oversample=20 max_expansion=20 gt=results/arxiv/gt.txt
```

### Step 5: Standalone Recall@K Evaluation

```bash
# C++ tool
./build/eval_recall --gt results/arxiv/gt.txt \
                    --pred results/arxiv/ann.txt \
                    --K 10 --verbose

# Python tool
python3 tools/eval_recall.py \
    --gt results/arxiv/gt.txt \
    --pred results/arxiv/ann.txt \
    --K 10
```

---

## 7. One-Command Experiment Scripts

### Full baseline comparison

```bash
DATASET=arxiv \
VEC=dataset/arxiv/vectors.fvecs \
STR=dataset/arxiv/strings.txt \
QRY=dataset/arxiv/queries.txt \
bash scripts/run_baselines.sh
```

Runs all methods (ground truth → RegExANN → pre-filter → post-filter → hierarchical) and prints a recall summary table. Key environment variables:

| Variable          | Default         | Description                                     |
| ----------------- | --------------- | ----------------------------------------------- |
| `EF_LIST`         | `10 20 50`      | `ef` values to sweep for `ann`                  |
| `OVERSAMPLE_LIST` | `10 20 50`      | `oversample` values to sweep for `postfilter`   |
| `MAX_EXPANSION`   | `0`             | `max_expansion` for postfilter (0 = unlimited)  |
| `SAMPLE_RATIOS`   | `1.0 0.5 0.2`   | `sample_ratio` values to sweep for `prefilter`  |
| `CLUSTERS`        | `100`           | Number of k-means clusters                      |
| `PQ_M`            | `8`             | PQ subspaces                                    |

### Single-dataset experiment

```bash
DATASET=arxiv \
VEC_FILE=dataset/arxiv/vectors.fvecs \
STR_FILE=dataset/arxiv/strings.txt \
QRY_FILE=dataset/arxiv/queries.txt \
bash scripts/run_experiment.sh
```

### Cluster count ablation

```bash
VEC_FILE=... STR_FILE=... QRY_FILE=... GT_FILE=... \
CLUSTER_COUNTS="10 25 50 100 200 400" \
bash scripts/sweep_clusters.sh
# Output: results/<dataset>/sweep_clusters.csv
```

### ef / sample_ratio / expansion sweep

```bash
VEC_FILE=... STR_FILE=... QRY_FILE=... GT_FILE=... \
bash scripts/sweep_params.sh
# Output: results/<dataset>/sweep_params.csv
```

---

## 8. Generating Query Files

```bash
python3 tools/gen_queries.py \
    --fvecs  dataset/arxiv/vectors.fvecs \
    --strings dataset/arxiv/strings.txt \
    --output  dataset/arxiv/queries.txt \
    --num_queries 1000 \
    --style wildcard \
    --seed 42
```

Supported `--style` values:

| Style         | Example              |
| ------------- | -------------------- |
| `substring`   | `neural`             |
| `prefix`      | `deep`               |
| `suffix`      | `search$`            |
| `alternation` | `graph\|network`     |
| `wildcard`    | `deep.*learning`     |
| `mixed`       | `ann.*gpu\|gpu.*ann` |

---

## 9. Dataset Preparation

```bash
# arXiv
python3 tools/prepare_dataset.py \
    --mode arxiv \
    --emb_file arxiv_embeddings.h5 \
    --str_file arxiv_titles.txt \
    --out_vecs dataset/arxiv/vectors.fvecs \
    --out_strs dataset/arxiv/strings.txt

# SIFT1M with randomly assigned DBLP titles
python3 tools/prepare_dataset.py \
    --mode sift \
    --emb_file sift/sift_base.fvecs \
    --str_file dblp/titles.txt \
    --out_vecs dataset/sift1m/vectors.fvecs \
    --out_strs dataset/sift1m/strings.txt

# General HDF5
python3 tools/prepare_dataset.py \
    --mode hdf5 \
    --emb_file laion1m.h5 \
    --vec_key emb --str_key caption \
    --out_vecs dataset/laion1m/vectors.fvecs \
    --out_strs dataset/laion1m/strings.txt
```

---

## 10. Hierarchical Index

Recommended for large-scale datasets (>1M vectors):

```bash
./build/regann \
    dataset/sift1m/vectors.fvecs \
    dataset/sift1m/strings.txt \
    dataset/sift1m/queries.txt \
    10 100 \
    results/sift1m/hier.txt \
    30 hier \
    k0=10 nprobe=5 pq_m=8 gt=results/sift1m/gt.txt
```

`k0` is the number of coarse clusters; `k1 = clusters / k0` is the number of fine clusters per coarse cluster.

---

## 11. Supported Regular Expression Syntax

The implementation uses C++ `std::regex` in case-insensitive mode.

```text
neural                  # substring
graph|network           # alternation
deep.*learning          # wildcard
(ann|knn).*search       # grouped alternation
[a-z]+network           # character class
^deep                   # anchor (prefix)
search$                 # anchor (suffix)
go+gle                  # quantifier +
colou?r                 # quantifier ?
```
