#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-all}"

# ── SIFT1M (HuggingFace mirror of the original texmex corpus) ─────────────
download_sift() {
    echo "════════════════════════════════════════════════════════════"
    echo " SIFT1M  (source: huggingface.co/datasets/qbo-odp/sift1m)"
    echo "════════════════════════════════════════════════════════════"
    mkdir -p sift
    wget -c -O sift/sift_base.fvecs \
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs"
    wget -c -O sift/sift_query.fvecs \
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs"
    wget -c -O sift/sift_groundtruth.ivecs \
        "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs"
    echo "→ sift/sift_base.fvecs ready"
}

# ── GIST1M (HuggingFace mirror, DiskANN fbin format -> converted to fvecs) ─
download_gist() {
    echo "════════════════════════════════════════════════════════════"
    echo " GIST1M  (source: huggingface.co/datasets/maknee/gist1m)"
    echo "════════════════════════════════════════════════════════════"
    mkdir -p gist
    wget -c -O gist/base.fbin \
        "https://huggingface.co/datasets/maknee/gist1m/resolve/main/fbin/base.fbin"

    echo "→ converting fbin -> fvecs ..."
    python3 - << 'PYEOF'
import numpy as np

IN_FBIN  = "gist/base.fbin"
OUT_FVECS = "gist/gist_base.fvecs"

with open(IN_FBIN, "rb") as f:
    npts, dim = np.fromfile(f, dtype=np.int32, count=2)
    print(f"fbin: npts={npts}, dim={dim}")
    vecs = np.fromfile(f, dtype=np.float32, count=int(npts) * int(dim))
    vecs = vecs.reshape(int(npts), int(dim))

with open(OUT_FVECS, "wb") as f:
    for v in vecs:
        f.write(np.int32(dim).tobytes())
        f.write(v.astype(np.float32).tobytes())

print(f"wrote {npts} vectors of dim {dim} -> {OUT_FVECS}")
PYEOF

    rm -f gist/base.fbin
    echo "→ gist/gist_base.fvecs ready"
}

# ── Msong + Audio (GQR datasets, CUHK) ─────────────────────────────────────
download_msong_audio() {
    echo "════════════════════════════════════════════════════════════"
    echo " Msong + Audio  (source: cse.cuhk.edu.hk/systems/hash/gqr)"
    echo "════════════════════════════════════════════════════════════"

    mkdir -p msong audio _gqr_tmp

    wget -c -O _gqr_tmp/msong.tar.gz \
        "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz"
    wget -c -O _gqr_tmp/audio.tar.gz \
        "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/audio.tar.gz"

    tar -xzf _gqr_tmp/msong.tar.gz -C _gqr_tmp
    tar -xzf _gqr_tmp/audio.tar.gz -C _gqr_tmp

    # The .fvecs filename inside the GQR tarball isn't guaranteed to match
    # msong_base.fvecs / audio_base.fvecs exactly, so locate it by pattern
    # and normalize the name instead of hardcoding the path.
    MSONG_SRC=$(find _gqr_tmp -iname '*msong*.fvecs' | head -1)
    AUDIO_SRC=$(find _gqr_tmp -iname '*audio*.fvecs' | head -1)

    if [[ -z "${MSONG_SRC}" ]]; then
        echo "[ERROR] Could not find a .fvecs file for msong inside the GQR archive."
        echo "        Check _gqr_tmp/ manually and adjust the find pattern above."
        exit 1
    fi
    if [[ -z "${AUDIO_SRC}" ]]; then
        echo "[ERROR] Could not find a .fvecs file for audio inside the GQR archive."
        echo "        Check _gqr_tmp/ manually and adjust the find pattern above."
        exit 1
    fi

    cp "${MSONG_SRC}" msong/msong_base.fvecs
    cp "${AUDIO_SRC}" audio/audio_base.fvecs

    rm -rf _gqr_tmp
    echo "→ msong/msong_base.fvecs ready (from ${MSONG_SRC})"
    echo "→ audio/audio_base.fvecs ready (from ${AUDIO_SRC})"
}


# ── DBLP title dump (used to assign literature titles to SIFT1M/GIST1M) ──
download_dblp() {
    echo "════════════════════════════════════════════════════════════"
    echo " DBLP XML dump  (source: dblp.org)"
    echo "════════════════════════════════════════════════════════════"
    mkdir -p dblp
    wget -c -O dblp/dblp.xml.gz \
        "https://dblp.org/xml/dblp.xml.gz"
    gunzip -f dblp/dblp.xml.gz
    echo "→ dblp/dblp.xml ready"
}

case "${TARGET}" in
    sift)  download_sift ;;
    gist)  download_gist ;;
    dblp)  download_dblp ;;
    msong) download_msong_audio ;;
    all)
        download_sift
        download_gist
        download_dblp
        download_msong_audio
        ;;
    *)
        echo "Usage: bash dataset/download_raw.sh [all|sift|gist|dblp|msong]"
        exit 1
        ;;
esac

echo ""
echo "Done."

