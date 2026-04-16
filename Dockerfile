FROM python:3.12-slim

# ── Build-time env vars ────────────────────────────────────────────────────────
# PIP_PROGRESS_BAR and PIP_NO_COLOR are REQUIRED on Docker 20.10.x servers.
# The default seccomp profile blocks the clone() syscall during builds, which
# causes pip's rich progress bar (thread-based) to crash with:
#   RuntimeError: can't start new thread
#
# NOTE: We do NOT call apt-get here. Docker 20.10's seccomp profile also
# blocks the syscalls used by apt's post-invoke cleanup scripts during build,
# causing "E: Problem executing scripts APT::Update::Post-Invoke" errors.
# --security-opt cannot be used during `docker build` on Docker 20.10 (only 23+).
# Modern PyTorch and sentence-transformers wheels bundle their own OpenMP
# runtime, so libgomp1 does not need to be installed from the system.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1
ENV RAG_CONFIG_PATH=/app/config_rag.yaml

WORKDIR /app

# ── Fix for pkg_resources issue ────────────────────────────────────────────────
# (REMOVED) PyMilvus 2.5+ no longer requires pkg_resources, so the setuptools
# workaround and diagnostic check is no longer needed.
RUN pip install --upgrade pip

# ── PyTorch: GTX 1080 (Pascal, sm_61) compatibility ──────────────────────────
# The GTX 1080 is a Pascal-architecture GPU (compute capability sm_61).
# PyTorch 2.3+ dropped sm_61 from prebuilt wheels — installing plain `torch`
# via pip gives the latest version which only supports sm_75+ (Turing and newer),
# producing: "NVIDIA GeForce GTX 1080 is not compatible with the current PyTorch"
#
# Fix: pin to torch==2.2.2 built against CUDA 11.8, which still includes sm_61.
# This must be installed BEFORE requirements.txt so it is not overridden.
RUN pip install --no-cache-dir \
    "torch==2.2.2+cu118" \
    "torchvision==0.17.2+cu118" \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ── NO apt-get calls in this Dockerfile ───────────────────────────────────────
# Docker 20.10.x's seccomp profile blocks the syscalls used by apt's post-invoke
# cleanup scripts during `docker build`. The error looks like:
#   E: Problem executing scripts APT::Update::Post-Invoke (exit code 100)
# `--security-opt` cannot be applied to `docker build` on Docker 20.10 (only 23+).
#
# OCR is handled by rapidocr-onnxruntime (pure Python/ONNX, no system packages
# needed) — already listed in requirements.txt. Tesseract is NOT required.

# Install remaining Python dependencies (torch and setuptools are already installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Force Headless OpenCV & Fix OCR ──────────────────────────────────────────
# This uninstalls any conflicting OpenCV versions and ensures only the 
# headless one exists. Pinned to 4.8.1.78 as 4.9+ has an upstream bug linking libGL.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
