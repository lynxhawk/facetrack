# Face Detection YOLO11

## 模型效果

### 4060ti 8G
| 模型 | 推理时间 | FPS | mAP50 |
|------|----------|-----|-------|
| yolov11n-face.pt | 13.9ms | 72 | 67.2% |
| yolov11s-face.pt | 80ms | 13 | 72.2% |
| yolov11m-face.pt | 160ms | 6 | 75.2% |

### AMD R7 5825U
| 模型 | 推理时间 | FPS | mAP50 |
|------|----------|-----|-------|
| yolov11n-face.pt | 85ms | 12 | 67.2% |
| yolov10n-face.pt | 78ms | 13 | - |
| yolov8n-face.pt | 70ms | 14 | - |
| yolov6n-face.pt | 110ms | 9 | - |

## 依赖安装
- `pip install ultralytics`
- `pip install -r requirements.txt`
- `pip install "numpy<2.0" --force-reinstall`

## 数据准备

- 下载WIDERFACE数据集 [WIDERFACE](http://shuoyang1213.me/WIDERFACE/):
- 下载预训练模型 [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) model
- 数据集转换脚本 `convert_wider_face.py`

## 数据集结构
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

## 训练模型
`yolo detect train data=./datasets/wider_face/wider_face.yaml model=yolov11n-face.pt epochs=100 imgsz=640`

## 验证模型
`yolo detect val model=yolov11s-face.pt data=./datasets/wider_face/wider_face.yaml`

## 预测
`yolo detect predict model=yolov11m-face.pt source=./datasets/WIDER_val/images/ save=True`
