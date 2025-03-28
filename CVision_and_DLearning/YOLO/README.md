
# 🧠 YOLOv8 Object Detection Project

This repository contains the full training, conversion, and deployment pipeline for an object detection system using **YOLOv8**.

## 📁 Project Structure

```
.
├── convert_model/                # Scripts to convert YOLO models (e.g., to ONNX, TensorRT)
├── data/                         # Custom dataset (after extraction from dataset.zip)
├── deploy/                       # Deployment-related code
├── extract_object_into_from_xml/ # Utility to convert XML annotations (Pascal VOC) to YOLO format
├── model/                        # Trained models, outputs, logs
├── dataset.zip                   # Zipped dataset file
├── yolov8m.pt                    # Trained YOLOv8m model
├── v1_training.ipynb             # First training notebook
├── v2_training.ipynb             # Improved/updated training notebook
├── test_demo_v8.jpg              # Example image for testing demo
├── link.txt                      # Contains link to dataset or model (if any)
├── README.md                     # This file
=======
---

## 📁 Repository Structure
```
.
├── 01_extract_object_into_from_xml.ipynb   # Convert XML to YOLO
├── yolo_training.ipynb                     # Train YOLOv8m
├── images/                                 # All images
│   ├── train/                              # Dữ liệu huấn luyện
│   ├── val/                                # Dữ liệu validation
│   └── test_demo_v8.jpg                    # Ảnh test mẫu
├── annotations/                            # VOC XML annotations
│   ├── train/
│   └── val/
├── labels/                                 # Output YOLO labels
│   ├── train/
│   └── val/
├── model/                                  # Chứa mô hình huấn luyện
│   ├── yolov8m.pt                          # Model YOLOv8 đã train
│   └── runs/                               # Thư mục chứa kết quả huấn luyện
├── deploy/                                 # Triển khai mô hình
├── convert_model/                          # Chuyển đổi mô hình (sang ONNX, TensorRT, v.v.)
├── data/                                   # Dữ liệu gốc
│   ├── dataset.zip                         # Dữ liệu nén
│   ├── data_test/                          # Thư mục chứa dữ liệu test
├── README.md                               # Mô tả dự án
├── link.txt                                # Liên kết tài liệu hoặc model
└── data.yaml                               # Dataset config for YOLO
```

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
>>>>>>> be777977b6c29289436eb7702b138410bdcc5883
```

## 🚀 Features

- Train custom object detection with YOLOv8
- Convert Pascal VOC (XML) to YOLO format
- Model format conversion (e.g., `.pt` → `.onnx`)
- Custom dataset handling and preparation
- Notebook-based training (v1 and v2)
- Demo image included for quick testing
- Model ready for deployment

## 🛠 Requirements

```bash
pip install ultralytics opencv-python matplotlib numpy
```

Or use a `requirements.txt` file if available.

## 🧪 Training

You can use the notebooks to train the model:

```bash
jupyter notebook v2_training.ipynb
```

Training uses `ultralytics` package with YOLOv8.

## 🧳 Dataset Preparation

- If your annotations are in Pascal VOC format (XML), use the notebook/code in `extract_object_into_from_xml/` to convert to YOLO format.
- Extract `dataset.zip` into the `data/` folder before training.

## 🧾 Conversion

To convert trained `.pt` model to ONNX or other formats, check scripts inside `convert_model/`.

## 🚀 Deployment

Deployment logic (e.g., via FastAPI, Flask, Triton, etc.) is available in `deploy/`.

## 🖼 Demo

You can use `test_demo_v8.jpg` and the trained model `yolov8m.pt` to perform inference.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model("test_demo_v8.jpg", show=True)
```

## 📄 License

This project is open source and free to use under the MIT License.
