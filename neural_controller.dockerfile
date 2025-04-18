FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment flags
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and essential packages
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip git wget curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python libraries
RUN pip install torch==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118 \
 && pip install \
    transformers==4.47.0 \
    datasets==3.1.0 \
    numpy==1.26.4 \
    tqdm \
    torchmetrics \
    scikit-learn

# Install RFM from GitHub
RUN git clone https://github.com/aradha/recursive_feature_machines.git /app/rfm \
 && pip install -e /app/rfm

# Optional: Copy local files into container
# COPY . /app

# Default to Python shell
CMD ["python"]
