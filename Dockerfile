FROM python:3.12-slim

# ── Build-time env vars ────────────────────────────────────────────────────────
# PIP_PROGRESS_BAR and PIP_NO_COLOR are REQUIRED on Docker 20.10.x servers.
# The default seccomp profile blocks the clone() syscall during builds, which
# causes pip's rich progress bar (thread-based) to crash with:
#   RuntimeError: can't start new thread
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1
ENV RAG_CONFIG_PATH=/app/config_rag.yaml

WORKDIR /app

# Install libgomp1 — required by OpenBLAS / FAISS which are used internally by
# sentence-transformers and PyTorch. Without this, numpy thread init crashes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# All packages (including fastapi, uvicorn) are now in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
