import os
import random
import re
import html
import numpy as np

# ====== 配置 ======
IN_FVECS = "gist/gist_base.fvecs"
DBLP_PATH = "dblp/dblp.xml"
OUT_TITLES = "gist/gist_titles_1m.txt"

TARGET_N = 1_000_000
SEED = 42
# ==================


def count_fvecs(fname):
    file_size = os.path.getsize(fname)

    with open(fname, "rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

    record_size = int(4 + dim * 4)
    n = int(file_size // record_size)

    if file_size % record_size != 0:
        raise ValueError(
            f"Invalid fvecs file: {fname}, "
            f"file_size={file_size}, record_size={record_size}, "
            f"remainder={file_size % record_size}"
        )

    return n, dim, record_size


def clean_title(s):
    if s is None:
        return ""

    s = html.unescape(s)
    s = str(s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s.strip()


def sample_dblp_titles(xml_path, target_n, seed):
    rng = random.Random(seed)

    reservoir = []
    seen = 0

    title_pattern = re.compile(r"<title>(.*?)</title>")

    print(f"Reading DBLP titles from {xml_path} ...")

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            m = title_pattern.search(line)
            if not m:
                continue

            title = clean_title(m.group(1))

            if not title:
                continue

            # 跳过 DBLP 首页那种非论文标题，可选
            if title.lower() in {"home page", "dblp"}:
                continue

            seen += 1

            # Reservoir sampling: 从所有 DBLP title 里面随机抽 target_n 个
            if len(reservoir) < target_n:
                reservoir.append(title)
            else:
                j = rng.randint(1, seen)
                if j <= target_n:
                    reservoir[j - 1] = title

            if seen % 500000 == 0:
                print(f"seen DBLP titles: {seen}")

    print("total DBLP titles seen:", seen)
    print("sampled titles:", len(reservoir))

    if len(reservoir) < target_n:
        raise ValueError(f"Not enough DBLP titles: {len(reservoir)} < {target_n}")

    rng.shuffle(reservoir)
    return reservoir


def write_titles(titles, out_file):
    print(f"Writing titles to {out_file} ...")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for i, title in enumerate(titles):
            f.write(title + "\n")

            if (i + 1) % 100000 == 0:
                print(f"written titles: {i + 1}/{len(titles)}")

    print("title writing done")


def main():
    if not os.path.exists(IN_FVECS):
        raise FileNotFoundError(f"Cannot find fvecs file: {IN_FVECS}")

    if not os.path.exists(DBLP_PATH):
        raise FileNotFoundError(f"Cannot find DBLP file: {DBLP_PATH}")

    n_vecs, dim, record_size = count_fvecs(IN_FVECS)

    print("Input vector file:", IN_FVECS)
    print("vectors:", n_vecs)
    print("dim:", dim)
    print("record size:", record_size)

    if dim != 960:
        print(f"Warning: GIST1M usually has dim=960, but got dim={dim}")

    if n_vecs < TARGET_N:
        raise ValueError(f"Not enough GIST vectors: {n_vecs} < {TARGET_N}")

    if n_vecs > TARGET_N:
        print(f"Warning: vector file has {n_vecs} vectors, but only generating {TARGET_N} titles")
        print("If you use all vectors, set TARGET_N = n_vecs")

    titles = sample_dblp_titles(DBLP_PATH, TARGET_N, SEED)
    write_titles(titles, OUT_TITLES)

    print("Done.")
    print(f"Vector file kept unchanged: {IN_FVECS}")
    print(f"Title output: {OUT_TITLES}")
    print(f"Title lines: {TARGET_N}")


if __name__ == "__main__":
    main()