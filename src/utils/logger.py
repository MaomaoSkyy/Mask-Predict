import csv
import sys
import time
from pathlib import Path


class Logger:
    """仅 rank 0 写入文件；其余 rank 走 no-op。"""

    def __init__(self, save_dir: str, rank: int = 0):
        self.rank = rank
        self.save_dir = Path(save_dir)
        if rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.save_dir / "train_log.csv"
            self._csv_init = not self.csv_path.exists()

    def info(self, msg: str):
        if self.rank != 0:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def log_metrics(self, step: int, metrics: dict):
        if self.rank != 0:
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self._csv_init:
                writer.writerow(["step"] + list(metrics.keys()))
                self._csv_init = False
            writer.writerow([step] + [f"{v:.6f}" if isinstance(v, float) else v
                                       for v in metrics.values()])
