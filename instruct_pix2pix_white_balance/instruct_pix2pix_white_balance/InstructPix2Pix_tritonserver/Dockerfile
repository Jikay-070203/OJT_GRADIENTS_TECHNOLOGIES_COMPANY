FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Set up folder work
WORKDIR /app

# Install pip and some basic tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if any
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

# Copy source code into container
COPY . .

# Set environment variables for CUDA
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Port cho FastAPI
EXPOSE 8000

# server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

