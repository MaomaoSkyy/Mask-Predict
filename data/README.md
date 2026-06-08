# 数据放置说明

SMD（Server Machine Dataset）下载：
<https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>

放置结构：
```
data/SMD/
├── train/
│   ├── machine-1-1.txt
│   ├── machine-1-2.txt
│   └── ...
├── test/
│   └── machine-1-1.txt
└── test_label/
    └── machine-1-1.txt
```

每个 `.txt` 文件为逗号分隔的 38 列浮点数（每行一个时间步）。
`test_label` 中每行一个 0/1 标签。
