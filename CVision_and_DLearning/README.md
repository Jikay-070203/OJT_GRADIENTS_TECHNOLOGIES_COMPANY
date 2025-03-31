# 🧠 AI Projects Summary - PyTorch, YOLO, Transformers

This repository contains a diverse collection of AI and deep learning projects implemented using **PyTorch**, covering topics from foundational neural networks to advanced detection and Transformer models.

---

## 🔍 1. Object Detection with YOLOv8

- Custom training using **Ultralytics YOLOv8**.
- Supports annotation conversion, model export to ONNX/TorchScript/.
- Deployment-ready using FastAPI and Triton Inference Server.
- Example use cases: face detection, smart surveillance.

📁 Notebook: `v2_training.ipynb`  
📦 Folder: `convert_model/`, `deploy/`, `extract_object_into_from_xml/`

---

## 🧠 2. VGG16 Image Classification

- Fine-tuning **pretrained VGG16** from `torchvision.models`.
- Adapt final layers for custom datasets.
- Visualize training, validate accuracy.
- Suitable for small to medium-scale image classification tasks.

📁 Notebook: `VGG16.ipynb`

---

## 🔁 3. Transformer from Scratch

- Manual implementation of Transformer architecture (no `nn.Transformer`).
- Learn attention, positional encoding, encoder/decoder blocks.
- Build sequence-to-sequence model for toy translation tasks.
- Great for educational understanding of Transformers.

📁 Notebook: `Transformer_From_Scratch.ipynb`

---

## 🔥 4. PyTorch Foundations & CV Series

A complete notebook series to build up PyTorch expertise step-by-step:

| Notebook                           | Topic                         |
| ---------------------------------- | ----------------------------- |
| `00_pytorch_fundamentals.ipynb`    | Tensor basics                 |
| `01_PyTorch_workflow.ipynb`        | Manual training pipeline      |
| `02_pytorch_classification.ipynb`  | CNN with FashionMNIST         |
| `03_pytorch_computer_vision.ipynb` | Transfer learning with ResNet |
| `04_pytorch_custom_datasets.ipynb` | Handling custom datasets      |

---

## 🛠 Requirements

```bash
pip install torch torchvision ultralytics matplotlib numpy
```

---

## 🎯 Use Cases

- Face recognition
- Image classification with transfer learning
- Educational deep dive into Transformers
- Hands-on PyTorch training from beginner to advanced

---

## 📄 License

Open-source educational projects for AI exploration and deployment.
