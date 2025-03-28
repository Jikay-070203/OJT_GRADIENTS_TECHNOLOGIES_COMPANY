# So sánh mô hình YOLOv8: `best.pt` vs `best.torchscript`

## 🔹 `best.pt` – Mô hình PyTorch gốc
- **Định dạng**: `.pt` (PyTorch)
- **Dùng để**: Huấn luyện (`train`), đánh giá (`val`), dự đoán (`predict`) bằng YOLOv8
- **Yêu cầu**: Cần môi trường có PyTorch + Ultralytics YOLO
- **Đặc điểm**:
  - Gồm toàn bộ kiến trúc mô hình + trọng số
  - Có thể huấn luyện tiếp
  - Có thể convert sang các định dạng khác như ONNX, TorchScript, TFLite, OpenVINO,...
- **Thích hợp khi**: Bạn đang phát triển, training hoặc đánh giá mô hình

---

## 🔹 `best.torchscript` – Mô hình TorchScript đã "đóng băng"
- **Định dạng**: `.torchscript` (TorchScript – dạng serialized model của PyTorch)
- **Dùng để**: Deploy mô hình nhanh trong app Python hoặc môi trường C++ (PyTorch runtime)
- **Yêu cầu**: PyTorch runtime (không cần YOLOv8 CLI)
- **Đặc điểm**:
  - Không thể huấn luyện tiếp
  - Đã được convert thành biểu đồ tính toán cố định (graph)
  - Chạy nhanh, nhẹ, không phụ thuộc Ultralytics
- **Thích hợp khi**: Bạn muốn nhúng mô hình vào app, không cần chỉnh sửa hoặc training lại