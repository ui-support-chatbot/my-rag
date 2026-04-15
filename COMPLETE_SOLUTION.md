# Complete Solution for pkg_resources Error in RAG System

## Problem Statement
The RAG system fails to start with the error: `ModuleNotFoundError: No module named 'pkg_resources'` when deployed on a research server with Docker 20.10.8.

## Root Cause Analysis
The error occurs because:
1. pymilvus requires pkg_resources module which comes from setuptools
2. Python 3.12-slim images don't include setuptools by default
3. Even though setuptools is in requirements.txt, the installation order matters
4. The container may be using cached layers without the fix

## Complete Solution

### 1. Updated Dockerfile
The Dockerfile has been modified to ensure setuptools is installed before other dependencies:

```dockerfile
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
# Install setuptools first to ensure pkg_resources is available before other packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir setuptools>=68.0

# ── Diagnostic: Verify pkg_resources availability ──────────────────────────────
RUN python -c "import pkg_resources; print('pkg_resources version:', pkg_resources.__version__)" || echo "pkg_resources not available"

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

# Install remaining Python dependencies (torch and setuptools are already installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Updated requirements.txt
Added explicit pkg-resources dependency:

```
# Core Dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0
pyyaml>=6.0
openai>=1.0.0
# setuptools provides pkg_resources, which pymilvus imports at startup.
# Python 3.12-slim no longer ships setuptools by default.
setuptools>=68.0

# Vector Store
pymilvus[milvus_lite]>=2.4.0,<2.5

# Embedding Models
sentence-transformers>=3.0.0
transformers>=4.51.0,<5.0
# torch is installed separately in the Dockerfile with a specific CUDA 11.8 build
# to ensure GTX 1080 (Pascal, sm_61) compatibility. Do not add torch here.
# numpy 2.x breaks the C ABI used by torch 2.2.2 (_ARRAY_API not found).
# Pin to 1.x to maintain compatibility.
numpy>=1.24.0,<2.0

# Parsing
docling
trafilatura
pymupdf
beautifulsoup4

# Evaluation
ragas
datasets
pandas

# Utils
tqdm

# Fix for pkg_resources issue
pkg-resources==0.0.0
```

## Step-by-Step Deployment Instructions

### On Your Server:

1. **Stop current containers:**
```bash
docker-compose down
```

2. **Remove old images to force rebuild:**
```bash
docker rmi my-rag-api
# Optionally clean all unused images
docker image prune -a
```

3. **Copy the updated files to your server:**
   - Replace your current `Dockerfile` with the updated version above
   - Replace your current `requirements.txt` with the updated version above

4. **Build fresh images (important: use --no-cache):**
```bash
docker-compose build --no-cache
```

5. **Start the services:**
```bash
docker-compose up -d
```

6. **Monitor the logs to confirm successful startup:**
```bash
docker-compose logs -f rag-api
```

You should now see the RAG pipeline initializing successfully instead of the pkg_resources error.

## Verification

After deployment:

1. **Check health endpoint:**
```bash
curl http://localhost:8000/health
```

2. **Verify logs show successful initialization:**
```bash
docker-compose logs rag-api | grep "RAG Pipeline initialized"
```

3. **Test a simple query to ensure everything works:**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "test"}'
```

## Additional Notes

- The diagnostic line in the Dockerfile will show during build if pkg_resources is available
- The --no-cache flag is essential to ensure the new Dockerfile changes take effect
- Both the Dockerfile and requirements.txt changes are necessary for the complete fix
- The fix maintains all existing functionality including GTX 1080 GPU compatibility