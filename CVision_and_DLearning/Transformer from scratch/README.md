# 🤖 Transformer From Scratch - PyTorch

This notebook builds a **Transformer model from scratch** using PyTorch, without relying on high-level modules like `nn.Transformer`. It's designed for those who want to deeply understand attention mechanisms, positional encoding, encoder/decoder structure, and the training loop.

---

## 📌 Key Contents

### 1. **Embeddings**

- Word embedding and positional encoding.
- Combine them into input embeddings.

### 2. **Scaled Dot-Product Attention**

- Compute attention from Q, K, V matrices.
- Apply softmax and optional masking (e.g., for decoder's future masking).

### 3. **Multi-Head Attention**

- Split QKV into multiple "heads" to capture different representations.
- Concatenate and project the results.

### 4. **Feedforward & LayerNorm**

- Two-layer MLP with Dropout.
- Residual connections and Layer Normalization.

### 5. **Encoder & Decoder Blocks**

- Stack multiple encoder and decoder layers.
- Use both self-attention and cross-attention mechanisms.

### 6. **Training**

- Use a toy dataset (e.g., number-to-word translation).
- Train the seq2seq Transformer model.
- Handle padding masks, shifted targets, and loss computation.

---

## 🧠 Learning Objectives

- Gain deep understanding of **attention mechanisms**.
- Learn to implement a full Transformer model manually.
- Apply to NLP tasks such as translation, summarization, and text generation.

---

## ⚙️ Requirements

- Python 3.7+
- PyTorch ≥ 1.10
- NumPy
- Matplotlib
