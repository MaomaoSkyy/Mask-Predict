"""从 raw SMD（28 台 txt）拼出 AnomalyTransformer 要的 SMD_{train,test,test_label}.npy。

AT 的 SMDSegLoader 读单条拼好的数组（StandardScaler 在它内部做，这里只拼原始值）。
按标准机器顺序(1-1..1-8, 2-1..2-9, 3-1..3-11)拼接 train/test/label，保证 label 与 test 对齐。

用法（在服务器上跑）：
  python scripts/make_at_smd.py \
      --smd_root /home/nick/projects/new/data/SMD \
      --out_dir  ~/projects/Anomaly-Transformer/dataset/SMD
"""
import argparse
from pathlib import Path

import numpy as np

MACHINES = ([f"machine-1-{i}" for i in range(1, 9)] +
            [f"machine-2-{i}" for i in range(1, 10)] +
            [f"machine-3-{i}" for i in range(1, 12)])  # 28 台，标准顺序


def _load(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smd_root", required=True, help="raw SMD 根目录(含 train/ test/ test_label/)")
    ap.add_argument("--out_dir", required=True, help="输出目录(AT 的 dataset/SMD)")
    args = ap.parse_args()

    root = Path(args.smd_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr, te, lb = [], [], []
    for m in MACHINES:
        a = _load(root / "train" / f"{m}.txt")
        b = _load(root / "test" / f"{m}.txt")
        c = _load(root / "test_label" / f"{m}.txt")
        tr.append(np.atleast_2d(a)); te.append(np.atleast_2d(b)); lb.append(np.atleast_1d(c))
        print(f"{m}: train{a.shape} test{b.shape} label{c.shape}")

    SMD_train = np.concatenate(tr, axis=0).astype(np.float32)
    SMD_test = np.concatenate(te, axis=0).astype(np.float32)
    SMD_test_label = np.concatenate(lb, axis=0).astype(np.float32)
    assert SMD_test.shape[0] == SMD_test_label.shape[0], "test 与 label 长度不一致!"

    np.save(out / "SMD_train.npy", SMD_train)
    np.save(out / "SMD_test.npy", SMD_test)
    np.save(out / "SMD_test_label.npy", SMD_test_label)
    print(f"\n[done] -> {out}")
    print(f"  SMD_train {SMD_train.shape}  SMD_test {SMD_test.shape}  "
          f"label {SMD_test_label.shape}  anom={SMD_test_label.mean()*100:.2f}%")


if __name__ == "__main__":
    main()
