# RegExANN

RegExANN is a regex-aware filtered Approximate Nearest Neighbor (ANN) search
engine. It combines k-means vector clustering, a cluster-level trigram
inverted index, and Product Quantization (PQ) to prune vector partitions
using safe trigram evidence extracted from a regex predicate, before
running approximate vector search and verifying the original regex on the
surviving candidates.

The design targets a gap in existing filtered-ANN work: most prior systems
support numeric range filters or categorical equality filters, but not
free-form regular-expression predicates over text attributes. RegExANN
treats the cluster-level trigram index as the interface between symbolic
regex filtering and geometric ANN partition selection, so that regex
predicates prune vector clusters *before* PQ scanning rather than only
filtering objects before or after ANN search. Cluster-level pruning never
produces false negatives — every object returned is verified against the
original regex — but it can retain false-positive clusters, which is an
intentional precision/recall trade-off discussed in the paper.

This repository contains the C++ implementation, the Python dataset
preparation pipeline, and the experiment scripts used to reproduce the
paper's results across eight datasets (arXiv, Words, DBpedia, SIFT1M,
GIST1M, LAION1M, Msong, Audio).

---

## 1. Repository Layout

```text
RegEXANN/
├── exp/                      C++ core (index, search, CLI)
│   ├── include/
│   ├── src/
│   └── CMakeLists.txt
├── dataset/                  Python dataset acquisition + preparation
│   ├── download_raw.sh       Downloads raw vectors (SIFT1M, GIST1M, Msong, Audio, DBLP)
│   ├── arxiv.py / dbpedia.py / gist.py / laion.py / sift.py / song.py / words.py
│   ├── gen_query.py          Selectivity-controlled query + ground-truth generator
│   └── gen_query_strict_style.py
├── scripts/
│   ├── datasets/              run_<dataset>.sh for all 8 datasets
│   ├── pgvector/               Pgvector (HNSW) baseline
│   ├── run_all.sh              Runs all 8 datasets sequentially
│   ├── run_sift_cluster_ef_sweep.sh
│   ├── run_sift_pattern_classes.sh
│   └── gen_queries_all.sh
```

---

## 2. Environment Requirements

| Tool          | Minimum Version | Purpose                              |
| ------------- | --------------- | ------------------------------------ |
| g++ / clang++ | C++17 support   | Compiling `exp/`                     |
| cmake         | 3.10+           | Build system                         |
| python3       | 3.9+            | Dataset preparation, auxiliary tools |

Ubuntu / Debian:

```bash
sudo apt-get install -y build-essential cmake python3 python3-pip wget
```

Python dependencies (only needed for `dataset/` scripts and the Pgvector
baseline; the C++ core has no Python dependency):

```bash
pip install -r requirements.txt
```

---

## 3. Compilation

```bash
cd exp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build outputs:
#   build/regann        Main program for search and index construction
#   build/eval_recall    Standalone Recall@K evaluation tool
```

---

## 4. Data Format

### 4.1 Vector File `.fvecs`

Each vector is stored as:

```text
[int32: dim][float32 × dim]
```

The `bvecs` format (`uint8` vectors) is also supported via `fmt=bvecs`.

### 4.2 String File `strings.txt`

One string per line, 1-to-1 with the vector file:

```text
deep learning for image retrieval
approximate nearest neighbor search
graph neural network embedding
...
```

### 4.3 Query File `query.txt`

One query per line: a regular expression, then a space, then the query
vector components:

```text
deep.*learning 0.123 -0.456 0.789 ...
neural|network 0.001 0.234 -0.567 ...
(ann|knn).*gpu 0.111 0.222 0.333 ...
```

---

## 5. Dataset Preparation

Each of the eight datasets needs three things in `dataset/<name>/`:
`vectors.fvecs`, `strings.txt`, and `query.txt` (plus a `groundtruth.txt`
generated separately — see §6).

### 5.1 Download raw source files

```bash
bash dataset/download_raw.sh          # SIFT1M, GIST1M, Msong, Audio, DBLP
bash dataset/download_raw.sh sift     # individual targets also work:
bash dataset/download_raw.sh gist     #   sift | gist | msong | dblp
```

This downloads the large binary files that are intentionally **not**
committed to the repo. Sources:

| Dataset       | Source                                                       |
| ------------- | ------------------------------------------------------------ |
| SIFT1M        | `huggingface.co/datasets/qbo-odp/sift1m`                     |
| GIST1M        | `huggingface.co/datasets/maknee/gist1m` (fbin, auto-converted to fvecs) |
| Msong / Audio | `cse.cuhk.edu.hk/systems/hash/gqr` (GQR datasets)            |
| DBLP titles   | `dblp.org/xml/dblp.xml.gz`                                   |
| arXiv         | pulled automatically via `datasets.load_dataset()`           |
| DBpedia       | pulled automatically via `datasets.load_dataset()`           |
| LAION-1M      | streamed directly via the `hf://` URI in `lance.dataset()` — no download needed |
| Words         | computed on the fly (`wordfreq` + `sentence-transformers`) — no download needed |

### 5.2 Run the per-dataset preparation script

```bash
python3 dataset/arxiv.py     # -> arxiv_vectors_all.fvecs, arxiv_titles_all.txt
python3 dataset/dbpedia.py   # -> dataset/dbpedia/{vectors.fvecs,strings.txt,query.txt,query_vectors.fvecs}
python3 dataset/gist.py      # -> gist/gist_titles_1m.txt (vectors stay at gist/gist_base.fvecs)
python3 dataset/sift.py      # -> sift/sift_titles_1m.txt (vectors stay at sift/sift_base.fvecs)
python3 dataset/laion.py     # -> laion_base.fvecs, laion_captions.txt
python3 dataset/song.py      # -> msong/msong_titles.txt, audio/audio_titles.txt
python3 dataset/words.py     # -> base.fvecs, base_words.txt
```

`dataset/dbpedia.py` is the only script that writes directly into the final
`dataset/<name>/{vectors.fvecs,strings.txt}` layout. The other six write
their vectors/titles under loosely-named local paths (matching each raw
source's own naming), so a final assembly step is needed:

```bash
mkdir -p dataset/arxiv && mv arxiv_vectors_all.fvecs dataset/arxiv/vectors.fvecs && mv arxiv_titles_all.txt dataset/arxiv/strings.txt
mkdir -p dataset/gist  && cp gist/gist_base.fvecs dataset/gist/vectors.fvecs && cp gist/gist_titles_1m.txt dataset/gist/strings.txt
mkdir -p dataset/sift  && cp sift/sift_base.fvecs dataset/sift/vectors.fvecs && cp sift/sift_titles_1m.txt dataset/sift/strings.txt
mkdir -p dataset/laion && mv laion_base.fvecs dataset/laion/vectors.fvecs && mv laion_captions.txt dataset/laion/strings.txt
mkdir -p dataset/msong && cp msong/msong_base.fvecs dataset/msong/vectors.fvecs && cp msong/msong_titles.txt dataset/msong/strings.txt
mkdir -p dataset/audio && cp audio/audio_base.fvecs dataset/audio/vectors.fvecs && cp audio/audio_titles.txt dataset/audio/strings.txt
mkdir -p dataset/words && mv base.fvecs dataset/words/vectors.fvecs && mv base_words.txt dataset/words/strings.txt
```

All textual fields are normalized through an inlined `clean_text()` helper
in each script (letters + spaces only, HTML entities unescaped first where
relevant, e.g. DBLP-sourced titles).

---

## 6. Generating Query + Ground-Truth Files

```bash
python3 dataset/gen_query.py \
    dataset/arxiv/strings.txt \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/query.txt \
    dataset/arxiv/groundtruth.txt \
    --n_queries 1000 --topk 10 --style mixed --seed 42 \
    --min_selectivity 0.01 --max_selectivity 0.10
```

`--style` accepts: `substring`, `prefix`, `suffix`, `alternation`,
`wildcard`, `mixed`, or `all` (cycles evenly through every style).

| Style         | Example              |
| ------------- | -------------------- |
| `substring`   | `neural`             |
| `prefix`      | `^deep`              |
| `suffix`      | `search$`            |
| `alternation` | `graph\|network`     |
| `wildcard`    | `deep.*learning`     |
| `mixed`       | `ann.*gpu\|gpu.*ann` |

`dataset/gen_query_strict_style.py` takes the same four positional
arguments and produces a stricter variant used for the regex
pattern-class experiment (§9). `scripts/gen_queries_all.sh` wraps
`gen_query.py` to generate queries for every dataset in one call.

---

## 7. Main Program Usage

```bash
./exp/build/regann <vectors.fvecs> <strings.txt> <queries.txt> <K> \
               <clusters> <output.txt> <max_iter> <algorithm> [options...]
```

| Argument        | Description                                                |
| --------------- | ---------------------------------------------------------- |
| `vectors.fvecs` | Dataset vector file                                        |
| `strings.txt`   | Dataset string file, corresponding to the vectors          |
| `queries.txt`   | Query file                                                 |
| `K`             | Number of nearest neighbors to return                      |
| `clusters`      | Number of k-means clusters (recommended: `50`–`500`)       |
| `output.txt`    | Output file. Each line stores the result IDs for one query |
| `max_iter`      | Maximum k-means iterations (recommended: `30`)             |
| `algorithm`     | Algorithm option — see §7.1                                |

### 7.1 Algorithm Options

| Algorithm     | Description                                              |
| ------------- | -------------------------------------------------------- |
| `ann`         | **RegExANN**: k-means + cluster-level trigram index + PQ |
| `groundtruth` | Exact full scan, used to generate ground truth           |
| `prefilter`   | Pre-filtering baseline: regex filter first, then kNN     |
| `postfilter`  | Post-filtering baseline: kNN first, then regex filter    |

### 7.2 Optional Parameters

All options use `key=value` format.

| Option             | Applies to    | Default          | Description                                                  |
| ------------------ | ------------- | ---------------- | ------------------------------------------------------------ |
| `pq_m=N`           | `ann`, `hier` | `8`              | Number of PQ subspaces (must divide vector dim)              |
| `pq_ksub=N`        | `ann`, `hier` | `256`            | Codebook size per PQ subspace                                |
| `ef=N`             | `ann`         | `K`              | Candidate pool size (≥ K). Larger → higher recall, slower query |
| `k0=N`             | `hier`        | `sqrt(clusters)` | Number of coarse clusters                                    |
| `nprobe=N`         | `hier`        | `k0/2`           | Number of coarse clusters probed                             |
| `oversample=N`     | `postfilter`  | `10`             | Initial candidate window is `K × N`                          |
| `max_expansion=N`  | `postfilter`  | `0`              | Window doubles until K matches found or `K × N` reached. `0` = no cap |
| `sample_ratio=F`   | `prefilter`   | `1.0`            | Fraction of the regex-matching set to search (0 < F ≤ 1.0)   |
| `gt=<file>`        | All           | None             | Ground truth file; if given, Recall@K is printed automatically |
| `save=<prefix>`    | `ann`         | None             | Save the index as `<prefix>.{kmidx,gramidx,pqidx}`           |
| `load=<prefix>`    | `ann`         | None             | Skip index construction and load a saved index               |
| `fmt=fvecs\|bvecs` | All           | `fvecs`          | Vector file format                                           |

---

## 8. Recall / Speed Trade-off Parameters

### 8.1 `ann` — `ef` (candidate pool size)

```bash
for EF in 10 20 50 100 200; do
    ./exp/build/regann ... ann ef=${EF} gt=gt.txt
done
```

Typical guidance: start at `ef=2K`, increase until recall meets your target.

### 8.2 `postfilter` — adaptive expansion

```bash
# Fixed window (original behaviour)
./exp/build/regann ... postfilter oversample=10 max_expansion=10 gt=gt.txt

# Adaptive: expand until K matches found or full scan
./exp/build/regann ... postfilter oversample=10 max_expansion=0 gt=gt.txt
```

### 8.3 `prefilter` — `sample_ratio`

```bash
for R in 1.0 0.5 0.2 0.1; do
    ./exp/build/regann ... prefilter sample_ratio=${R} gt=gt.txt
done
```

---

## 9. Experiment Scripts

```bash
# Run all 8 datasets sequentially (ef sweep, prefilter sweep, postfilter sweep
# for each, writing results/<dataset>/summary.csv with the 4-stage pipeline
# breakdown built in)
bash scripts/run_all.sh

# A single dataset
bash scripts/datasets/run_arxiv.sh

# Cluster-count ablation (Figure 4d in the paper)
bash scripts/run_sift_cluster_ef_sweep.sh

# Regex pattern-class breakdown (Figure 4a in the paper)
bash scripts/run_sift_pattern_classes.sh

# Pgvector (HNSW) baseline — requires a running PostgreSQL + pgvector instance
bash scripts/pgvector/run_pg_hnsw_all.sh
```

`results/<dataset>/summary.csv` already includes the per-stage timing
breakdown (`t1_trigram_parse_ms`, `t2_cluster_lookup_ms`, `t3_pq_scan_ms`,
`t4_regex_verify_ms`, `t5_rerank_ms`) for every `ann` row, so no separate
profiling pass is required. To build a cross-dataset table, concatenate the
`method=ann` rows from the per-dataset CSVs, e.g.:

```python
import pandas as pd, glob
dfs = []
for f in glob.glob("results/*/summary.csv"):
    d = pd.read_csv(f)
    d["dataset"] = f.split("/")[1]
    dfs.append(d[d.method == "ann"])
pd.concat(dfs).to_csv("results/profile.csv", index=False)
```

## 10. Supported Regular Expression Syntax

The implementation uses C++ `std::regex` in case-insensitive mode.

```text
neural                  # substring
graph|network           # alternation
deep.*learning          # wildcard
(ann|knn).*search       # grouped alternation
[a-z]+network            # character class
^deep                    # anchor (prefix)
search$                  # anchor (suffix)
go+gle                   # quantifier +
colou?r                  # quantifier ?
```

---

## 11. Citation

```bibtex
@article{regexann2027,
  title   = {RegExANN: Efficient Approximate Nearest Neighbor Search with Regular Expression Filtering},
  author  = {Zhong, Shurui and Ye, Weitang and Luo, Siqiang},
  volume  = {20},
  number  = {1},
  year    = {2027}
}
```