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

# Install Python dependencies
# All packages (including fastapi, uvicorn) are now in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
