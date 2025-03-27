# 🤖 Transformer From Scratch - PyTorch

Notebook này xây dựng mô hình **Transformer hoàn chỉnh từ đầu** bằng PyTorch, không dùng các hàm có sẵn như `nn.Transformer`. Phù hợp cho người muốn hiểu sâu cơ chế attention, positional encoding, encoder/decoder và training loop.

---

## 📌 Nội dung chính

### 1. **Embeddings**

- Word embedding và positional encoding.
- Kết hợp thành input embeddings.

### 2. **Scaled Dot-Product Attention**

- Tính toán attention từ Q, K, V.
- Softmax với mask (tự động che đi tương lai trong decoder).

### 3. **Multi-Head Attention**

- Tách QKV thành nhiều "đầu" để học đa chiều.
- Nối lại và chiếu về đầu ra.

### 4. **Feedforward & LayerNorm**

- MLP hai tầng + Dropout.
- Residual connections + LayerNorm.

### 5. **Encoder & Decoder Block**

- Stack nhiều layer encoder và decoder.
- Dùng self-attention và cross-attention.

### 6. **Training**

- Dataset toy: dịch đơn giản (ví dụ: từ số sang chữ).
- Huấn luyện mô hình dịch seq2seq.
- Mask padding, target shifting, loss.

---

## 🧠 Mục tiêu học được:

- Hiểu sâu **cơ chế attention**.
- Tự viết mô hình Transformer không phụ thuộc thư viện ngoài.
- Áp dụng vào các bài toán NLP như dịch, tóm tắt, sinh văn bản.

---

## ⚙️ Yêu cầu:

- Python 3.7+
- PyTorch ≥ 1.10
- NumPy, Matplotlib
