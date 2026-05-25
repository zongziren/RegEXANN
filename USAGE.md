# RegExANN — 编译与运行指南

## 1. 环境要求

| 工具 | 最低版本 | 用途 |
|------|----------|------|
| g++ / clang++ | C++17 支持 | 编译 |
| cmake | 3.10+ | 构建系统 |
| python3 | 3.7+ | 辅助工具（可选） |

Ubuntu / Debian 一键安装：
```bash
sudo apt-get install -y build-essential cmake
```

---

## 2. 编译

```bash
# 进入项目根目录
cd RegExANN

# 创建构建目录并编译（Release 模式，-O3 优化）
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 编译产物
#   build/regann        主程序（搜索 + 索引构建）
#   build/eval_recall   独立 Recall@K 评估工具
```

---

## 3. 数据格式

### 3.1 向量文件（.fvecs）
标准 fvecs 二进制格式，每个向量：
```
[int32: dim][float32 × dim]
```
也支持 bvecs 格式（uint8 向量），使用 `fmt=bvecs` 选项。

### 3.2 字符串文件（strings.txt）
每行一条字符串，与向量文件**逐行对应**：
```
deep learning for image retrieval
approximate nearest neighbor search
graph neural network embedding
...
```

### 3.3 查询文件（queries.txt）
每行一条查询：正则表达式 + 空格 + 向量分量（空格分隔）：
```
deep.*learning 0.123 -0.456 0.789 ...
neural|network 0.001 0.234 -0.567 ...
(ann|knn).*gpu 0.111 0.222 0.333 ...
```

---

## 4. 主程序用法

```
./build/regann <vectors.fvecs> <strings.txt> <queries.txt> <K>
               <clusters> <output.txt> <max_iter> <algorithm> [options...]
```

| 参数 | 说明 |
|------|------|
| `vectors.fvecs` | 数据集向量文件 |
| `strings.txt` | 数据集字符串文件（与向量对应） |
| `queries.txt` | 查询文件 |
| `K` | 返回最近邻数量 |
| `clusters` | k-means 聚类数（推荐 50~500） |
| `output.txt` | 输出文件（每行为一条查询的结果 ID） |
| `max_iter` | k-means 最大迭代次数（推荐 30） |
| `algorithm` | 见下表 |

### 4.1 算法选项

| algorithm | 说明 |
|-----------|------|
| `ann` | **RegExANN**（论文主方法：k-means + trigram + PQ） |
| `hier` | 两级层次 RegExANN |
| `groundtruth` | 精确全扫描（生成 ground truth） |
| `baseline` | 同 groundtruth |
| `prefilter` | Pre-filtering 基线（先过滤再 kNN） |
| `postfilter` | Post-filtering 基线（先 kNN 再过滤） |

### 4.2 可选参数（key=value 格式）

| 选项 | 适用算法 | 默认值 | 说明 |
|------|----------|--------|------|
| `pq_m=N` | ann, hier | 8 | PQ 子空间数（必须整除向量维度） |
| `pq_ksub=N` | ann, hier | 256 | 每个子空间的码本大小 |
| `k0=N` | hier | √clusters | 粗聚类数 |
| `nprobe=N` | hier | k0/2 | 查询时探测的粗聚类数 |
| `oversample=N` | postfilter | 10 | 过采样倍数 |
| `gt=<file>` | 所有 | — | ground truth 文件，自动计算并打印 Recall@K |
| `save=<prefix>` | ann | — | 构建后保存索引到 `<prefix>.{kmidx,gramidx,pqidx}` |
| `load=<prefix>` | ann | — | 跳过构建，直接加载已保存的索引 |
| `fmt=fvecs\|bvecs` | 所有 | fvecs | 向量文件格式 |

---

## 5. 典型工作流

### 步骤 1：生成 Ground Truth（精确结果，用于评估）

```bash
./build/regann \
    dataset/arxiv/vectors.fvecs \
    dataset/arxiv/strings.txt \
    dataset/arxiv/queries.txt \
    10 100 \
    results/arxiv/gt.txt \
    30 groundtruth
```

### 步骤 2：运行 RegExANN，保存索引

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

输出示例：
```
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

### 步骤 3：下次直接加载索引（跳过构建）

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

### 步骤 4：运行基线对比

```bash
# Pre-filter
./build/regann ... 30 prefilter gt=results/arxiv/gt.txt

# Post-filter（过采样 20 倍）
./build/regann ... 30 postfilter oversample=20 gt=results/arxiv/gt.txt
```

### 步骤 5：独立 Recall@K 评估

```bash
# C++ 工具
./build/eval_recall --gt results/arxiv/gt.txt \
                    --pred results/arxiv/ann.txt \
                    --K 10 --verbose

# Python 工具（同样功能）
python3 tools/eval_recall.py \
    --gt results/arxiv/gt.txt \
    --pred results/arxiv/ann.txt \
    --K 10
```

---

## 6. 一键实验脚本

```bash
# 设置环境变量后一键运行全套实验
export DATASET=arxiv
export VEC_FILE=dataset/arxiv/vectors.fvecs
export STR_FILE=dataset/arxiv/strings.txt
export QRY_FILE=dataset/arxiv/queries.txt
export K=10
export CLUSTERS=100
export BIN=./build/regann

bash scripts/run_experiment.sh
# 依次执行：ground truth → RegExANN → pre-filter → post-filter → 打印 Recall 对比
```

消融实验（不同 cluster 数对 QPS/Recall 的影响）：

```bash
export CLUSTER_COUNTS="10 25 50 100 200 400"
bash scripts/sweep_clusters.sh
# 输出：results/arxiv/sweep_clusters.csv
```

---

## 7. 生成查询文件

```bash
python3 tools/gen_queries.py \
    --fvecs  dataset/arxiv/vectors.fvecs \
    --strings dataset/arxiv/strings.txt \
    --output  dataset/arxiv/queries.txt \
    --num_queries 1000 \
    --style wildcard \
    --seed 42
```

支持的正则模式（`--style`）：

| 模式 | 示例 |
|------|------|
| `substring` | `neural` |
| `prefix` | `deep` |
| `suffix` | `search$` |
| `alternation` | `graph\|network` |
| `wildcard` | `deep.*learning` |
| `mixed` | `ann.*gpu\|gpu.*ann` |

---

## 8. 数据集准备

```bash
# arXiv（HDF5 嵌入 + 标题文本）
python3 tools/prepare_dataset.py \
    --mode arxiv \
    --emb_file arxiv_embeddings.h5 \
    --str_file arxiv_titles.txt \
    --out_vecs dataset/arxiv/vectors.fvecs \
    --out_strs dataset/arxiv/strings.txt

# SIFT1M（fvecs + DBLP 随机字符串）
python3 tools/prepare_dataset.py \
    --mode sift \
    --emb_file sift/sift_base.fvecs \
    --str_file dblp/titles.txt \
    --out_vecs dataset/sift1m/vectors.fvecs \
    --out_strs dataset/sift1m/strings.txt

# 通用 HDF5
python3 tools/prepare_dataset.py \
    --mode hdf5 \
    --emb_file laion1m.h5 \
    --vec_key emb --str_key caption \
    --out_vecs dataset/laion1m/vectors.fvecs \
    --out_strs dataset/laion1m/strings.txt
```

---

## 9. 层次索引（大规模数据集推荐）

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

`k0`（粗聚类数）× `k1`（细聚类数）≈ `clusters`，减少查询时的质心比较次数。

---

## 10. 正则表达式语法支持

支持标准 C++ `std::regex` 的全部语法（icase 模式，大小写不敏感）：

```
# 子串匹配
neural

# 交替（OR）
graph|network

# 通配符
deep.*learning

# 分组交替
(ann|knn).*search

# 字符类
[a-z]+network

# 锚点
^deep
search$

# 量词
go+gle
colou?r
```

---

## 11. 常见问题

**Q: `pq_m` 设置失败？**  
A: `pq_m` 必须整除向量维度。程序会自动向下调整，日志中有提示。

**Q: 召回率偏低？**  
A: 增大 `clusters` 数量，或增大 `pq_ksub`（码本更精细）。对于 `postfilter`，增大 `oversample`。

**Q: 内存不足？**  
A: 减小 `clusters` 或 `pq_ksub`，或使用 `hier` 模式减少内存峰值。

**Q: 想跳过每次重建索引？**  
A: 第一次运行加 `save=idx/mydata`，后续运行加 `load=idx/mydata`，直接复用。
