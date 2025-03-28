# 🚀 YOLOv8m Training Pipeline from VOC XML Annotations

This repository provides a two-step pipeline for training YOLOv8m on a custom dataset annotated in Pascal VOC XML format. The pipeline includes:

1. 📦 Converting `.xml` annotations to YOLOv8 format
2. 🧠 Training YOLOv8m with the Ultralytics framework

---

## 📁 Repository Structure

```
.
├── 01_extract_object_into_from_xml.ipynb   # Convert XML to YOLO
├── yolo_training.ipynb                     # Train YOLOv8m
├── images/                                 # All images
│   ├── train/
│   └── val/
├── annotations/                            # VOC XML annotations
│   ├── train/
│   └── val/
├── labels/                                 # Output YOLO labels
│   ├── train/
│   └── val/
└── data.yaml                               # Dataset config for YOLO
```

---

## 🔧 Step 1: Convert XML Annotations to YOLOv8 Format

Open and run **`01_extract_object_into_from_xml.ipynb`**.

### 🔄 What it does:
- Parses VOC-style XML files
- Extracts: filename, size, object (label + bounding box)
- Converts to YOLO format: `class_id x_center y_center width height` (normalized)
- Saves `.txt` labels with the same name as image

### ⚙️ Before running, edit:

```python
folder_path = "annotations/train"
output_folder = "labels/train"
images_path = "images/train"
label_dict = {"your_class_name": 0}  # Mapping label → index
```

> 📌 Lặp lại tương tự cho tập `val`.

---

## 🧠 Step 2: Train YOLOv8m Model

Open and run **`yolo_training.ipynb`**.

### ✅ What it does:
- Loads YOLOv8m with `YOLO("yolov8m.pt")`
- Trains using the specified `data.yaml`
- Outputs:
  - Training logs
  - ✅ Best model weights: `runs/detect/train/weights/best.pt`

### 🔧 Training parameters:

```python
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## 📤 Step 3: Export Trained Model (Optional)

Sau khi huấn luyện, bạn có thể export mô hình sang nhiều định dạng phổ biến để triển khai trên các nền tảng khác nhau.

### ✅ Các định dạng đã export:

| Format         | File/Folder Output                          |
|----------------|---------------------------------------------|
| ONNX           | `best.onnx`                                 |
| TorchScript    | `best.torchscript`                          |
| OpenVINO       | `best_openvino_model/`                      |
| TFLite         | `best_float32.tflite`, `best_full_integer_quant.tflite` |

### 🧪 Test thử sau khi export

Ví dụ: test nhanh mô hình ONNX sau khi export:

```python
import onnxruntime
import cv2
import numpy as np

ort_session = onnxruntime.InferenceSession("best.onnx")
img = cv2.imread("test.jpg")
# Tiền xử lý ảnh, resize, normalize (tùy theo input model)
# Sau đó thực hiện inference:
outputs = ort_session.run(None, {"images": input_tensor})
```

> 📌 Tương tự, bạn có thể dùng mô hình TorchScript, OpenVINO, hoặc TFLite tùy theo nền tảng triển khai (mobile, edge, cloud...).

---

## ✅ Notes

- YOLOv8 không cần định dạng `.cfg`, chỉ cần `.yaml` và cấu trúc đúng.
- Hãy đảm bảo ảnh và nhãn có **cùng tên**, ví dụ: `image1.jpg` ↔ `image1.txt`.
- Tất cả giá trị trong label YOLO phải **normalized** từ 0 → 1.
- Với TFLite export, bạn có thể tạo mô hình:
  - Float32
  - Full integer quantization (để dùng trên microcontroller hoặc thiết bị ràng buộc tài nguyên)

---

## ✍️ Author

Script developed by [Your Name]  
Feel free to raise issues or contribute!