# Face Detection YOLO11

### 模型效果
| 模型 | 推理时间 | FPS | mAP50 |
|------|----------|-----|-------|
| yolo11n-face.pt | 40ms | 25 | 67.2% |
| yolo11s-face.pt | 80ms | 13 | 72.2% |
| yolo11m-face.pt | 160ms | 6 | 75.2% |

### 数据准备

- 下载WIDERFACE数据集 [WIDERFACE](http://shuoyang1213.me/WIDERFACE/):
- 下载预训练模型 [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) model
- 数据集转换脚本 `convert_wider_face.py`

### 数据集结构
```
datasets/wider_face/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── wider_face.yaml
```

### 训练模型
`yolo detect train data=./datasets/wider_face/wider_face.yaml model=yolov11n-face.pt epochs=100 imgsz=640`

### 验证模型
`yolo detect val model=yolov11s-face.pt data=./datasets/wider_face/wider_face.yaml`

### 预测
`yolo detect predict model=yolov11m-face.pt source=./datasets/WIDER_val/images/ save=True`
