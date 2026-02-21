FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python (devel image needed because CuPy compiles kernels at runtime via nvcc)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# CACHEBUST arg invalidates the COPY layer cache on each deploy,
# ensuring CUDA kernel source changes are always picked up.
# Layers above this (apt-get, pip install) remain cached.
ARG CACHEBUST=0
RUN mkdir -p server
COPY . server/

EXPOSE 8420
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8420", "--workers", "1"]
