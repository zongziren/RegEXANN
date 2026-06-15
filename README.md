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
# Enter the project root directory
cd RegExANN

# Create the build directory and compile in Release mode with -O3 optimization
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

The `bvecs` format, which stores `uint8` vectors, is also supported by using the `fmt=bvecs` option.

### 3.2 String File `strings.txt`

Each line contains one string. The strings must correspond to the vector file line by line.

```text
deep learning for image retrieval
approximate nearest neighbor search
graph neural network embedding
...
```

### 3.3 Query File `queries.txt`

Each line contains one query: a regular expression followed by a space and then the vector components separated by spaces.

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
| `ann`         | **RegExANN**, the main method in the paper: k-means + trigram + PQ |
| `hier`        | Two-level hierarchical RegExANN                                    |
| `groundtruth` | Exact full scan, used to generate ground truth                     |
| `baseline`    | Same as `groundtruth`                                              |
| `prefilter`   | Pre-filtering baseline: filter first, then run kNN                 |
| `postfilter`  | Post-filtering baseline: run kNN first, then filter                |

### 4.2 Optional Parameters

Optional parameters follow the `key=value` format.

| Option             | Applicable Algorithms | Default          | Description                                                                    |
| ------------------ | --------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `pq_m=N`           | `ann`, `hier`         | `8`              | Number of PQ subspaces. It must divide the vector dimension                    |
| `pq_ksub=N`        | `ann`, `hier`         | `256`            | Codebook size for each PQ subspace                                             |
| `k0=N`             | `hier`                | `sqrt(clusters)` | Number of coarse clusters                                                      |
| `nprobe=N`         | `hier`                | `k0/2`           | Number of coarse clusters to probe during search                               |
| `oversample=N`     | `postfilter`          | `10`             | Oversampling factor                                                            |
| `gt=<file>`        | All                   | None             | Ground truth file. If provided, Recall@K is automatically computed and printed |
| `save=<prefix>`    | `ann`                 | None             | Save the constructed index as `<prefix>.{kmidx,gramidx,pqidx}`                 |
| `load=<prefix>`    | `ann`                 | None             | Skip index construction and directly load a saved index                        |
| `fmt=fvecs\|bvecs` | All                   | `fvecs`          | Vector file format                                                             |

---

## 5. Typical Workflow

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
    pq_m=8 save=results/arxiv/idx/arxiv gt=results/arxiv/gt.txt
```

Example output:

```text
[INFO] Dataset: 132687 vectors (dim=768), 132687 strings.
[INFO] Building index (k=100, pq_m=8, pq_ksub=256) …
[INFO] K-means done.
[INFO] Trigram index: 3821 trigrams.
[INFO] Training PQ (m=8, k_sub=256) …
[INFO] Index built in 4231 ms.
[INFO] Memory: 312.5 MB
[INFO] Queries          : 1000
[INFO] Avg total time   : 2.14 ms
[INFO] Avg set-op time  : 0.08 ms
[INFO] Avg cluster time : 2.05 ms
[INFO] QPS              : 467.3
[EVAL] Recall@10 = 91.4 %
```

### Step 3: Load the Index Next Time

```bash
./build/regann \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/strings.txt \
    dataset/arxiv/queries.txt \
    10 100 \
    results/arxiv/ann2.txt \
    30 ann \
    load=results/arxiv/idx/arxiv gt=results/arxiv/gt.txt
```

### Step 4: Run Baseline Methods

```bash
# Pre-filtering
./build/regann ... 30 prefilter gt=results/arxiv/gt.txt

# Post-filtering with an oversampling factor of 20
./build/regann ... 30 postfilter oversample=20 gt=results/arxiv/gt.txt
```

### Step 5: Standalone Recall@K Evaluation

```bash
# C++ tool
./build/eval_recall --gt results/arxiv/gt.txt \
                    --pred results/arxiv/ann.txt \
                    --K 10 --verbose

# Python tool with the same functionality
python3 tools/eval_recall.py \
    --gt results/arxiv/gt.txt \
    --pred results/arxiv/ann.txt \
    --K 10
```

---

## 6. One-Command Experiment Script

```bash
# Set environment variables and run the full experiment pipeline
export DATASET=arxiv
export VEC_FILE=dataset/arxiv/vectors.fvecs
export STR_FILE=dataset/arxiv/strings.txt
export QRY_FILE=dataset/arxiv/queries.txt
export K=10
export CLUSTERS=100
export BIN=./build/regann

bash scripts/run_experiment.sh
# This runs: ground truth → RegExANN → pre-filtering → post-filtering → Recall comparison
```

Ablation experiment on the impact of different numbers of clusters on QPS and Recall:

```bash
export CLUSTER_COUNTS="10 25 50 100 200 400"
bash scripts/sweep_clusters.sh
# Output: results/arxiv/sweep_clusters.csv
```

---

## 7. Generating Query Files

```bash
python3 tools/gen_queries.py \
    --fvecs  dataset/arxiv/vectors.fvecs \
    --strings dataset/arxiv/strings.txt \
    --output  dataset/arxiv/queries.txt \
    --num_queries 1000 \
    --style wildcard \
    --seed 42
```

Supported regular expression styles specified by `--style`:

| Style         | Example              |
| ------------- | -------------------- |
| `substring`   | `neural`             |
| `prefix`      | `deep`               |
| `suffix`      | `search$`            |
| `alternation` | `graph\|network`     |
| `wildcard`    | `deep.*learning`     |
| `mixed`       | `ann.*gpu\|gpu.*ann` |

---

## 8. Dataset Preparation

```bash
# arXiv: HDF5 embeddings + title text
python3 tools/prepare_dataset.py \
    --mode arxiv \
    --emb_file arxiv_embeddings.h5 \
    --str_file arxiv_titles.txt \
    --out_vecs dataset/arxiv/vectors.fvecs \
    --out_strs dataset/arxiv/strings.txt

# SIFT1M: fvecs + randomly assigned DBLP titles
python3 tools/prepare_dataset.py \
    --mode sift \
    --emb_file sift/sift_base.fvecs \
    --str_file dblp/titles.txt \
    --out_vecs dataset/sift1m/vectors.fvecs \
    --out_strs dataset/sift1m/strings.txt

# General HDF5 dataset
python3 tools/prepare_dataset.py \
    --mode hdf5 \
    --emb_file laion1m.h5 \
    --vec_key emb --str_key caption \
    --out_vecs dataset/laion1m/vectors.fvecs \
    --out_strs dataset/laion1m/strings.txt
```

---

## 9. Hierarchical Index

Recommended for large-scale datasets:

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

Here, `k0` is the number of coarse clusters and `k1` is the number of fine clusters. Their product is approximately equal to `clusters`. This design reduces the number of centroid comparisons during query processing.

---

## 10. Supported Regular Expression Syntax

The implementation supports the full syntax of C++ `std::regex`, using case-insensitive mode.

```text
# Substring matching
neural

# Alternation / OR
graph|network

# Wildcard
deep.*learning

# Grouped alternation
(ann|knn).*search

# Character class
[a-z]+network

# Anchors
^deep
search$

# Quantifiers
go+gle
colou?r
```
