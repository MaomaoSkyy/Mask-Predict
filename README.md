# Mask-Predict for Time-Series Anomaly Detection

Transformer encoder + BERT 风格 Mask-then-Predict，用于多变量时序异常检测（首要数据集：SMD）。

详细设计与开发规范见 [CLAUDE.md](CLAUDE.md)。

## Quick Start

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
下载 SMD 后按如下结构放置：
```
data/SMD/
├── train/machine-1-1.txt
├── test/machine-1-1.txt
└── test_label/machine-1-1.txt
```
来源：<https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>

### 3. 训练（2×3090 DDP）
```bash
bash scripts/train_ddp.sh configs/smd.yaml machine-1-1
```

### 4. 评估
```bash
python scripts/eval.py --config configs/smd.yaml --entity machine-1-1 --ckpt runs/machine-1-1/best.pt
```
