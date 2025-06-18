# 🧠 InstructPix2Pix Triton API

FastAPI-based inference server for **InstructPix2Pix** using ONNX models deployed via **NVIDIA Triton Inference Server**. Supports both **CPU** and **GPU** runtime. Designed for scalable, production-ready deployment (Docker).

---

## 🚀 Features

- 🖼️ Image-to-image editing using `InstructPix2Pix`
- ⚡ Fast inference with ONNX models + Triton Server
- 🔁 Supports classifier-free guidance + image guidance
- ☁️ Ready for cloud: Docker, Compose, Kubernetes, Helm
- 📤 API ready: `POST /inference`

---

## 🏗️ Project Structure

<pre>
.
├── app/                    # FastAPI app source code
│   ├── app.py              # API Endpoint
│   └── triton_clients/     # Triton gRPC clients: vae_encoder, unet, vae_decoder
├── Dockerfile              # Docker build config
├── docker-compose.yml      # Multi-container deployment (CPU/GPU)
├── requirements.txt        # Python dependencies
│   ├── triton_clients/     # Triton gRPC or HTTP clients for model inference
│   │   ├── vae_encoder_client.py   # VAE Encoder client
│   │   ├── vae_decoder_client.py   # VAE Decoder client
│   │   ├── unet_client.py          # UNet client
├── charts/                 # Helm chart for Kubernetes deployment
│   └── instructpix2pix/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
├── k8s/                    # Kubernetes manifests
│   ├── instructpix2pix-deploy.yaml
│   └── instructpix2pix-service.yaml
└── README.md              
</pre>

---

## 🐳 setup with Docker

```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -vD:path\model:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models

uvicorn app:app --host 0.0.0.0 --port 8080 --reload

-Build:
docker build -t instructpix2pix-triton .
docker run --gpus all -p 8000:8000 instructpix2pix-triton
```
