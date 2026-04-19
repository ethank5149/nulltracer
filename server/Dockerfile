FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY nulltracer/ nulltracer/

RUN pip3 install --no-cache-dir .

COPY server/requirements.txt server/requirements.txt
RUN pip3 install --no-cache-dir -r server/requirements.txt

COPY server/ server/

EXPOSE 8420
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8420", "--workers", "1"]