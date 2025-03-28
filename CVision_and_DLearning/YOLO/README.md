
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
