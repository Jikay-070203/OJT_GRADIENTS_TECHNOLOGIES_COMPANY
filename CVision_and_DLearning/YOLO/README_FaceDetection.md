# 😊 Face Detection with YOLOv5 + OpenCV DNN

Dự án này huấn luyện mô hình **YOLOv5 nhận diện khuôn mặt** từ dữ liệu gán nhãn định dạng XML (Pascal VOC), xuất model sang định dạng `.onnx`, và sử dụng OpenCV để thực hiện nhận diện realtime hoặc từ ảnh tĩnh.

---

## 📁 Nội dung file

### 📌 `01_extract_object_into_from_xml.ipynb`
- Trích xuất nhãn khuôn mặt từ file `.xml` (Pascal VOC).
- Chuyển về định dạng YOLO (`.txt`) để huấn luyện.
- Lưu label vào thư mục `labels/`.

### 📌 `yolo_training.ipynb`
- Clone repo YOLOv5 và cấu hình huấn luyện nhận diện khuôn mặt.
- Sử dụng `data.yaml` để chỉ định đường dẫn ảnh, nhãn, số class.
- Train mô hình YOLOv5 với dữ liệu khuôn mặt.
- Xuất model sang định dạng `.onnx` để dùng với OpenCV.

### 📌 `yolo_predictions.py`
- Class `YOLO_Pred` dùng để:
  - Load mô hình YOLO `.onnx`.
  - Thực hiện dự đoán khuôn mặt từ ảnh đầu vào.
  - Áp dụng Non-Maximum Suppression (NMS).
  - Vẽ bounding box, tên class và độ chính xác.
- Sử dụng OpenCV DNN, không cần PyTorch khi chạy dự đoán.

---

## 🧪 Ứng dụng
- Nhận diện khuôn mặt trong ảnh tĩnh hoặc video.
- Nhẹ, nhanh, có thể tích hợp web/app/mobile.
- Thích hợp triển khai trên thiết bị biên (Edge AI).

---

## ⚙️ Yêu cầu
- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch (chỉ dùng để huấn luyện)
- NumPy, PyYAML

---

## 🚀 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Chạy notebook sau để chuyển XML → YOLO format:
```bash
01_extract_object_into_from_xml.ipynb
```

### 2. Huấn luyện mô hình nhận diện khuôn mặt
```bash
yolo_training.ipynb
```

### 3. Dự đoán khuôn mặt từ ảnh
```python
from yolo_predictions import YOLO_Pred
face_detector = YOLO_Pred("best.onnx", "data.yaml")
img = cv2.imread("face.jpg")
result = face_detector.predictions(img)
cv2.imshow("Face Detection", result)
```

---

## 📝 Ghi chú
- Class hỗ trợ nhiều khuôn mặt trong ảnh.
- Có thể kết hợp webcam realtime.
- Model `.onnx` dùng được cả trên OpenCV, OpenVINO,...
