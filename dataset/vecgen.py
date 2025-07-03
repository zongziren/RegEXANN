import numpy as np
from pathlib import Path


SRC_PATH = "./SIFT10K/sift10k.fvecs"   
DST_PATH = "./SIFT100/sift100.fvecs"
NUM_TO_KEEP = 100


def copy_first_n_fvecs(src_path: str, dst_path: str, n: int):
    src = Path(src_path)
    if not src.exists():
        print(f"❌ 未找到源文件: {src_path}\n请先将源 .fvecs 文件上传到工作目录后，再重新运行此代码。")
        return
    
    with src.open("rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        bytes_per_vec = 4 + 4 * dim 
        total_bytes = bytes_per_vec * n
        
        data = np.fromfile(f, dtype=np.uint8, count=total_bytes)
    
    with open(dst_path, "wb") as out:
        out.write(data.tobytes())
    
copy_first_n_fvecs(SRC_PATH, DST_PATH, NUM_TO_KEEP)


