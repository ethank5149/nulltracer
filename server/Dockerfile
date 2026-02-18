FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and OpenGL libs (OpenGL kept temporarily for A/B validation)
RUN apt-get update && apt-get install -y python3 python3-pip libegl1 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p server
COPY . server/

# Tell PyOpenGL to use EGL for context management (not GLX/GLUT).
# Without this, PyOpenGL's contextdata module can't find the EGL context
# and raises "Attempt to retrieve context when no valid context".
ENV PYOPENGL_PLATFORM=egl

EXPOSE 8420
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8420", "--workers", "1"]
