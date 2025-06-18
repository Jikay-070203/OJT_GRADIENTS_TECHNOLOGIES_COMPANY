---

# 🧠 AI Model Collection: Training, Deployment & Applications

Dự án này là bộ sưu tập các project về **Computer Vision** và **Deep Learning** với các chủ đề:
- Huấn luyện mô hình (PyTorch, YOLO, VGG16, Transformer)
- Chuyển đổi và triển khai mô hình (ONNX, Triton Inference Server, Trism, FastAPI)
- Ứng dụng thực tế: Nhận diện đối tượng, phân loại ảnh, chỉnh sửa ảnh dựa trên văn bản (InstructPix2Pix), v.v.

## 📂 Cấu trúc chính

### 1. `CVision_and_DLearning/`
- **YOLO**: Huấn luyện, chuyển đổi, triển khai YOLOv8 cho nhận diện khuôn mặt, đối tượng.
- **VGG16**: Fine-tune VGG16 cho phân loại ảnh.
- **Transformer from scratch**: Xây dựng Transformer thủ công để học về NLP.
- **PyTorch notebook series**: Học PyTorch từ cơ bản đến nâng cao.

### 2. `Deploy_model/`
- **Task 1**: Chuyển đổi mô hình sang ONNX, super resolution, phục vụ với Triton.
- **Task 2**: Triển khai mô hình với Triton, benchmark hiệu năng.
- **Task 3**: FastAPI gọi Triton qua gRPC/HTTP.
- **Task 4**: FastAPI + Trism, tự động tải model từ Hugging Face khi khởi động container.

### 3. `instruct_pix2pix_white_balance/`
- **InstructPix2Pix Triton API**: Triển khai server inference cho InstructPix2Pix (chỉnh sửa ảnh theo văn bản) với ONNX + Triton + FastAPI, hỗ trợ Docker/Kubernetes.
- **Ví dụ chạy local, colab**: Hướng dẫn và script chạy thử nghiệm.

---

## 🛠 Công nghệ sử dụng

- PyTorch, TorchVision, Ultralytics YOLO
- ONNX, Triton Inference Server, Trism
- FastAPI, Uvicorn
- Docker, Kubernetes, Helm
- Hugging Face Model Hub

---

## 🚀 Hướng dẫn cài đặt nhanh

```bash
pip install torch torchvision ultralytics matplotlib numpy fastapi uvicorn onnx tritonclient
```

---

## 📄 CUSTOM LICENSE

Copyright (c) 2025 Nguyen Thanh Hoa

Bạn được phép sử dụng, sao chép, chỉnh sửa, chia sẻ, thương mại hóa dự án này với điều kiện:

- **Ghi rõ nguồn tác giả:** Nguyen Thanh Hoa
- **Không được kiện cáo, phản bác hay gây ảnh hưởng tiêu cực đến tác giả**
- **Mọi thay đổi phải nêu rõ và giữ lại phần ghi công gốc**
- **Dự án được cung cấp "nguyên trạng", không có bất kỳ bảo đảm nào. Tác giả không chịu trách nhiệm cho mọi rủi ro phát sinh từ việc sử dụng.**

---
