
# 🧠 AI Model Collection: Training, Deployment & Applications

This repository is a collection of advanced deep learning projects that showcase model training, deployment, and application pipelines using **PyTorch**, **ONNX**, **Triton Inference Server**, and **FastAPI**. It includes foundational learning, custom training, and production-grade deployment scenarios.

---

## 📦 Project Highlights

### 🔍 1. Object Detection & Face Recognition (YOLOv8)

- Train YOLOv8 on custom datasets (e.g., trash detection, smart camera).
- Convert models to ONNX/TorchScript.
- Deploy using Triton Server + FastAPI.
- Track people & classify objects in real-time.

📂 `v2_training.ipynb`, `convert_model/`, `deploy/`

---

### 🖼 2. InstructPix2Pix API (Image Editing)

- Deploy ONNX models for image-to-image editing using text prompts.
- Components: VAE Encoder, UNet, VAE Decoder via Triton Inference Server.
- FastAPI backend for REST API `POST /inference`.
- Docker + Kubernetes ready.

📂 `app/`, `charts/`, `k8s/`, `triton_clients/`

---

### 🚀 3. Model Deployment Pipeline (Trism + HuggingFace)

- Convert and serve PyTorch/ONNX models.
- Deploy with Triton or **Trism** (lightweight wrapper).
- Download models automatically from HuggingFace at startup.
- Benchmark inference using Triton Performance Analyzer.
- FastAPI + gRPC/HTTP API endpoints.

📂 `Deploy_model/Task_1-4/`

---

### 🧠 4. Vision Classification with VGG16

- Fine-tune `torchvision.models.vgg16` for custom image classification.
- Modify classifier head, visualize metrics.
- Structured for easy experimentation.

📂 `VGG16.ipynb`

---

### 🔁 5. Transformer from Scratch

- Build a Transformer manually using PyTorch.
- Learn embeddings, attention, multi-head, encoder-decoder blocks.
- Use toy datasets (e.g., number to word) for training.
- For educational understanding of NLP models.

📂 `Transformer_From_Scratch.ipynb`

---

### 📘 6. PyTorch Notebook Series

- Learn PyTorch from basics to vision models.
- Tensor ops, training loop, CNN, ResNet, custom datasets.
- Great for students, educators, or ML beginners.

📂 `00_` → `04_pytorch_custom_datasets.ipynb`

---

## 🛠 Tech Stack

- PyTorch, TorchVision
- ONNX
- Triton Inference Server, Trism
- FastAPI, Uvicorn
- Docker, Kubernetes, Helm
- Hugging Face Model Hub

---

## 🧰 Requirements

```bash
pip install torch torchvision ultralytics matplotlib numpy fastapi uvicorn onnx tritonclient
```

---

## 📄 License

Open-source AI project collection for research, education, and deployment.
