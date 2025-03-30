
# 🔍 YOLOv8 Object Detection - Custom Training & Deployment

This project provides a full pipeline for object detection using **YOLOv8**, including training, conversion, deployment, and testing.

## 📁 Folder Structure

```
.
├── .ipynb_checkpoints/           # Auto-saved Jupyter checkpoints
├── convert_model/                # Scripts for model format conversion (e.g., TorchScript, ONNX)
├── data/                         # Training dataset (after unzipping)
├── deploy/                       # Deployment logic (e.g., FastAPI, Triton)
├── extract_object_into_from_xml/ # Convert PascalVOC XML annotations to YOLO format
├── model/                        # Output models, logs, results
├── dataset.zip                   # Compressed dataset
├── link.txt                      # Link to dataset or model (optional)
├── README.md                     # This file
├── test_demo_v8.jpg              # Image used to demo inference
├── v1_training.ipynb             # First version of training notebook
├── v2_training.ipynb             # Updated training notebook
├── yolov8m.pt                    # Trained YOLOv8m model
```

## 🚀 Features

- Train YOLOv8 with custom dataset
- Use XML annotations and convert to YOLO format
- Export trained model to `.pt`, TorchScript, ONNX
- Demo inference with sample image
- Ready for deployment via API

## 🛠 Requirements

```bash
pip install ultralytics opencv-python matplotlib numpy
```

## 🧪 How to Train

1. Prepare dataset (in `data/` or extract from `dataset.zip`)
2. Run `v2_training.ipynb` notebook:
   ```bash
   jupyter notebook v2_training.ipynb
   ```

3. Model will be saved as `yolov8m.pt` inside `model/` or root folder.

## 🧾 Convert Model

Inside `convert_model/`, you can export `.pt` to TorchScript or ONNX format.

## 🖼 Test with Sample Image

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model("test_demo_v8.jpg", show=True)
```

## 🚀 Deploy Model

Use code in `deploy/` to run the model via REST API or other backends.

## 📄 License

Open source for research and development.
