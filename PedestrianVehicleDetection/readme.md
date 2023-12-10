# 行人與車輛偵測

## Todo
- [X] 訓練行人與車輛偵測模型
- [X] 軌跡追蹤
- [ ] 車輛與行人分類
- [ ] 斑馬線狀態
- [ ] 違規偵測


## 專案目錄結構

```
.
├── main.py # 主程式
│
│
│
│
├── image
│   ├── image.png
│   └── zidane.jpg # IDK WTF is this
│
├── model
│   └── 存放模型
│
└── train # 行人與車輛 YOLOV8
    ├── VisDrone.yaml
    ├── runs
    │   └── detect
    │       └── train$
    │           ├── args.yaml
    │           └── weights
    ├── train.ipynb
    └── yolov8n.pt # base model
```
