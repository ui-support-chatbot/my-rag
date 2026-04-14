# 🚀 Server Deployment Guide — MyRAG on Docker 20.10.x

This guide covers the end-to-end deployment of MyRAG on your `riset-01` research server: Ubuntu, Docker 20.10.8, 2× NVIDIA GTX 1080, no `sudo` access.

> [!IMPORTANT]
> This guide is specifically written for **Docker version 20.10.x**. Several workarounds documented here do **not** apply to Docker 23+ and exist solely because of limitations in the older Docker daemon.

---

## Table of Contents

1. [Understanding the Docker 20.10 Security Problem](#1-understanding-the-docker-2010-security-problem)
2. [Prerequisites](#2-prerequisites)
3. [Clone & Prepare](#3-clone--prepare)
4. [The `docker-compose.yml` — Full Annotated Configuration](#4-the-docker-composeyml--full-annotated-configuration)
5. [Build the Image](#5-build-the-image)
6. [Initialize the Stack](#6-initialize-the-stack)
7. [Ingest Your Data](#7-ingest-your-data)
8. [Test & Verify](#8-test--verify)
9. [Ollama Setup & GPU Allocation](#9-ollama-setup--gpu-allocation)
10. [Persistence & Data Management](#10-persistence--data-management)
11. [Troubleshooting Reference](#11-troubleshooting-reference)

---

## 1. Understanding the Docker 20.10 Security Problem

This is the most important section. All the non-standard flags in our `docker-compose.yml` exist because of this.

Docker uses a **seccomp profile** (a Linux syscall filter) to restrict what containers can do. The default Docker seccomp profile blocks certain syscalls. In Docker **20.10.x**, this profile is more restrictive than in later versions, causing two specific failures:

### Problem 1: Build-time crash — `RuntimeError: can't start new thread`
- **When**: `docker build` running `pip install`
- **Why**: `pip`'s progress bar (powered by `rich`) spawns a background thread. Threading requires the `clone()` syscall. The Docker 20.10 seccomp profile blocks `clone()` during `docker build`.
- **Critical**: `docker build` in Docker 20.10 does **not** support `--security-opt`. You cannot relax the seccomp during build.
- **Our Fix**: `ENV PIP_PROGRESS_BAR=off` and `ENV PIP_NO_COLOR=1` in the Dockerfile eliminating all `rich`/thread usage by pip.

### Problem 2: Runtime crash — `OpenBLAS blas_thread_init: pthread_create failed`
- **When**: Container starts and Python imports `numpy` or any torch/sentence-transformers code
- **Why**: OpenBLAS (which backs numpy/torch) tries to spawn 16 threads for BLAS operations. This also uses `clone()`. At runtime, seccomp *can* be relaxed — but only via `--security-opt` on `docker run` or `security_opt:` in compose.
- **Our Fix**: `security_opt: [seccomp:unconfined]` and `pids_limit: -1` in `docker-compose.yml`.

---

## 2. Prerequisites

Verify these on the server before starting:

```bash
# 1. Docker version (should be 20.10.x)
docker --version

# 2. NVIDIA driver & GPU available
nvidia-smi

# 3. NVIDIA Container Toolkit (required for GPU passthrough)
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 4. Ollama running and accessible
curl http://localhost:11434/api/tags

# 5. A model pulled in Ollama
# If not, pull one:
docker exec -it ollama ollama pull qwen2.5:7b
```

**Model recommendations for 2× GTX 1080 (8 GB VRAM each):**

| Model | VRAM | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `qwen2.5:3b` | ~3 GB | Good | Fast | Best for testing |
| `qwen2.5:7b` | ~5 GB | Better | Medium | ✅ Recommended |
| `gemma3:4b` | ~4 GB | Better | Medium | High faithfulness |
| `qwen2.5:14b` | ~10 GB ⚠️ | Best | Slow | Needs both GPUs |

---

## 3. Clone & Prepare

```bash
# 1. Clone the repository
git clone https://github.com/aryakdaniswara/my-rag.git
cd my-rag

# 2. Pull latest updates (if already cloned)
git pull origin main

# 3. Create required directories for Milvus persistence and HuggingFace model cache
mkdir -p storage/etcd storage/minio storage/milvus storage/hf_cache

# 4. Place your PDF documents in the data/ directory
# (or copy them via scp from your local machine)
mkdir -p data
# Example: scp -r ./local_papers/* user@riset-01:~/my-rag/data/
```

---

## 4. The `docker-compose.yml` — Full Annotated Configuration

This is the complete `docker-compose.yml`. Every non-obvious setting is explained inline.

```yaml
version: '3.8'

services:

  # ── 1. etcd — Milvus metadata store ──────────────────────────────────────
  etcd:
    container_name: myrag-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_RETENTION_REASON=periodic
      - ETCD_AUTO_COMPACTION_RETENTION_RUNTIME=1h
    command: etcd
      -advertise-client-urls http://127.0.0.1:2379
      -listen-client-urls http://0.0.0.0:2379
      --data-dir /etcd
    volumes:
      - ./storage/etcd:/etcd
    restart: unless-stopped

  # ── 2. MinIO — Milvus object storage ─────────────────────────────────────
  minio:
    container_name: myrag-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    command: minio server /data
    volumes:
      - ./storage/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped

  # ── 3. Milvus Standalone — vector database ────────────────────────────────
  # We use Milvus Standalone (3-container stack) instead of milvus-lite because
  # milvus-lite's embedded SQLite engine panics under the Docker 20.10 seccomp
  # profile even with seccomp=unconfined. The standalone stack is more stable.
  milvus:
    container_name: myrag-milvus
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - ./storage/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - etcd
      - minio
    restart: unless-stopped

  # ── 4. MyRAG API — the main application ──────────────────────────────────
  rag-api:
    container_name: my-rag-api
    build: .
    ports:
      - "8000:8000"
    volumes:
      # Your documents (mounted read-only for safety)
      - ./data:/app/data:ro
      # Use the server config (connects to Milvus container, not localhost)
      - ./config_server.yaml:/app/config_rag.yaml
      # Cache HuggingFace model weights so they don't re-download on restart
      - ./storage/hf_cache:/root/.cache/huggingface
    environment:
      - RAG_CONFIG_PATH=/app/config_rag.yaml
      # Ollama (and many OpenAI-compatible endpoints) require an API key
      # variable to be set, even though it's not actually validated.
      - OPENAI_API_KEY=dummy

    # ── GPU Configuration ──────────────────────────────────────────────────
    # GPU 0: Dense (Harrier) + Sparse (OpenSearch) embedding models
    # GPU 1: Jina-v3 Reranker
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]

    # ── Docker 20.10 Security Workarounds ─────────────────────────────────
    #
    # WHY pids_limit: -1?
    #   Docker 20.10's default PID limit can prevent Python ML libraries
    #   (OpenBLAS, PyTorch) from spawning the threads they need for BLAS
    #   operations. Setting -1 removes the PID limit.
    #
    # WHY security_opt: seccomp:unconfined?
    #   This disables the seccomp syscall filter for this container.
    #   It allows clone(), mmap_min_addr, and other syscalls that
    #   OpenBLAS/numpy require to spawn BLAS worker threads.
    #   Without this, the container crashes on startup with:
    #     "OpenBLAS blas_thread_init: pthread_create failed"
    #   NOTE: This is safe for a research server environment. For production
    #   internet-facing deployments, use a custom seccomp profile instead.
    pids_limit: -1
    security_opt:
      - seccomp:unconfined

    # ── Network: Reaching Ollama on the host ──────────────────────────────
    # Ollama runs directly on the host (not in Docker), so the container
    # needs to reach the host's localhost. This maps host.docker.internal
    # to the host machine's IP, allowing:
    #   config_server.yaml:  llm_endpoint: "http://host.docker.internal:11434/v1"
    extra_hosts:
      - "host.docker.internal:host-gateway"

    depends_on:
      - milvus
    restart: unless-stopped

volumes:
  # Named volumes as fallback (local bind mounts in ./storage/ are preferred
  # for easier backup and inspection)
  etcd-data:
  minio-data:
  milvus-data:
  hf_cache:
```

> [!TIP]
> The `./storage/hf_cache` bind mount is critical — it persists downloaded model weights (~5-10 GB for Harrier + OpenSearch + Jina) across container restarts and rebuilds.

---

## 5. Build the Image

```bash
# DOCKER_BUILDKIT=0 is REQUIRED on Docker 20.10.x
# BuildKit (the default in Docker 23+) has known issues with the older daemon.
DOCKER_BUILDKIT=0 docker compose build
```

> [!WARNING]
> Do **not** use `docker compose build` without `DOCKER_BUILDKIT=0` on this server. BuildKit may fail silently or produce an invalid image on Docker 20.10.8.

If the build fails, check these first:
```bash
# Check if pip progress bar env vars are set in the image
docker compose build --no-cache 2>&1 | head -50

# Common error: "RuntimeError: can't start new thread"
# → Already fixed by PIP_PROGRESS_BAR=off in Dockerfile
```

---

## 6. Initialize the Stack

```bash
# 1. Start all services (Milvus Standalone + RAG API)
docker compose up -d

# 2. Watch startup logs — wait until you see "RAG Pipeline initialized"
docker compose logs -f rag-api

# 3. Verify Milvus is healthy before querying
docker compose logs milvus | tail -20

# 4. Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy","milvus":"connected"}
```

**Expected startup sequence:**
```
myrag-etcd    | ... etcd ready
myrag-minio   | ... MinIO server ready
myrag-milvus  | ... Milvus ready
my-rag-api    | INFO  - RAG Pipeline initialized from /app/config_rag.yaml
my-rag-api    | INFO  - Application startup complete.
```

---

## 7. Ingest Your Data

Once the stack is healthy, trigger document ingestion. This runs as a **background task** inside the existing container:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data"}'
```

**Expected response:**
```json
{
  "status": "ingestion_started",
  "directory": "/app/data",
  "message": "Check container logs for progress."
}
```

**Monitor progress:**
```bash
docker compose logs -f rag-api
# You'll see lines like:
# INFO  - Embedding 847 chunks (batch_size=32) ...
# INFO  - Indexed 847 chunks into 'documents'
# INFO  - Background ingestion completed for: /app/data
```

---

## 8. Test & Verify

### Basic query test
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa itu mekanisme penelaahan usulan pembukaan program studi?"
  }'
```

### Document-filtered query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Syarat pembukaan program studi baru",
    "metadata_filter": {"doc_id": "sk_rektor_001"}
  }'
```

### List indexed collections
```bash
curl http://localhost:8000/collections
```

### Swagger UI
Open `http://<SERVER_IP>:8000/docs` in your browser.

> [!NOTE]
> Port 8000 must be open on the server firewall. If you can't reach it externally, ask your server admin to run: `sudo ufw allow 8000/tcp`

---

## 9. Ollama Setup & GPU Allocation

Ollama should already be running on the host. To confirm it has GPU access and the right model:

```bash
# Check models available
curl http://localhost:11434/api/tags

# Test inference (should complete in a few seconds with GPU)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

**If Ollama is running on CPU only**, recreate it with GPU access:
```bash
docker rm -f ollama
docker run -d --name ollama \
  --gpus all \
  --privileged \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  --restart unless-stopped \
  ollama/ollama
```

**GPU allocation strategy (2× GTX 1080):**

| GPU | Used By | When |
|-----|---------|------|
| GPU 0 | Harrier dense embedding + OpenSearch sparse embedding | Ingestion & query time |
| GPU 1 | Jina-v3 reranker | Query time only |
| Host (Ollama) | LLM generation | Query time only |

> [!TIP]
> You can check GPU utilization during a query with `nvidia-smi` on the host.

---

## 10. Persistence & Data Management

All data is stored in `./storage/` on the host, which is bind-mounted into containers. You own these files and can back them up directly.

```
my-rag/
├── storage/
│   ├── etcd/       ← Milvus cluster metadata
│   ├── minio/      ← Milvus vector segments (binary)
│   ├── milvus/     ← Milvus runtime state
│   └── hf_cache/   ← HuggingFace model weights (~5-10 GB)
└── data/           ← Your source PDF/HTML documents
```

**To re-ingest after adding new documents:**
```bash
# Just POST to /ingest again — it appends, does not overwrite existing vectors
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data"}'
```

**To wipe the vector DB and start fresh:**
```bash
docker compose down
rm -rf ./storage/etcd ./storage/minio ./storage/milvus
mkdir -p storage/etcd storage/minio storage/milvus
docker compose up -d
# Then re-ingest
```

**Backup the vector DB:**
```bash
# Stop Milvus first to avoid partial reads
docker compose stop milvus minio etcd
tar -czf milvus_backup_$(date +%Y%m%d).tar.gz ./storage/
docker compose start etcd minio milvus
```

---

## 11. Troubleshooting Reference

This section documents every production issue encountered on `riset-01` and its fix.

### Build errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: can't start new thread` during `docker build` | pip's rich progress bar uses threads; Docker 20.10 seccomp blocks `clone()` during builds | `ENV PIP_PROGRESS_BAR=off` in Dockerfile ✅ (already set) |
| `E: Problem executing scripts APT::Update::Post-Invoke` (exit code 100) | Docker 20.10 seccomp profile blocks syscalls used by apt's post-invoke cleanup scripts during build. `--security-opt` cannot be used during `docker build` on Docker 20.10 (only Docker 23+) | Remove **all** `apt-get` calls from Dockerfile. Modern PyTorch/sentence-transformers pip wheels bundle their own OpenMP runtime — `libgomp1` from the system package is not needed ✅ (already fixed) |
| `Tokio executor failed: PermissionDenied` | `uv` package manager uses io_uring syscalls, blocked by seccomp during build | Use plain `pip` — never `uv` inside Docker on this server |
| `cannot stat 'storage': permission denied` during build | Milvus ran as root and created root-owned directories in the build context | Add `storage/` to `.dockerignore` ✅ (already set) |
| `No module named 'pkg_resources'` | `pymilvus` requires `setuptools`, which is missing from `python:3.12-slim` | Add `setuptools>=68.0` to `requirements.txt` ✅ (already fixed) |
| `Failed to initialize NumPy: _ARRAY_API not found` | NumPy 2.x broke C ABI compatibility with Torch 2.2.2 | Pin `numpy>=1.24.0,<2.0` in `requirements.txt` ✅ (already fixed) |
| `CollectionSchema.add_field() missing ... arguments` | Wrong keyword arguments passed to Milvus schema (`name` vs `field_name`) | Fixed in `milvus_client.py` ✅ (already fixed) |

### Runtime errors

| Error | Cause | Fix |
|-------|-------|-----|
| `OpenBLAS blas_thread_init: pthread_create failed` | numpy/torch's BLAS spawns 16 threads; seccomp blocks `clone()` at runtime | `security_opt: [seccomp:unconfined]` + `pids_limit: -1` ✅ (already in compose) |
| `RuntimeError: No CUDA GPUs are available` | Reranker loads on GPU but container has no GPU access | Ensure `deploy.resources.reservations.devices` is in compose ✅ |
| `Milvus Lite: file opened by another program` | milvus-lite SQLite lock — only one process can use it at a time | Use Milvus Standalone (3-container stack) instead of milvus-lite ✅ |
| `Connection refused` to Ollama from container | Wrong `llm_endpoint` in config_server.yaml | Use `http://host.docker.internal:11434/v1` (not `localhost`) ✅ (already in config_server.yaml) |
| `OPENAI_API_KEY not found` | Ollama requires the var set even though it's a dummy | `environment: OPENAI_API_KEY=dummy` ✅ (already in compose) |
| Port 8000 unreachable from network | Server firewall (`ufw`) blocking the port | Ask server admin: `sudo ufw allow 8000/tcp` |

### Performance

| Symptom | Cause | Fix |
|---------|-------|-----|
| First query takes 2-5 minutes | Models downloading on first use | Mount `./storage/hf_cache` and wait for initial download to complete once |
| Slow inference (minutes per query) | Ollama running on CPU, not GPU | `docker run --gpus all ollama/ollama` (see Section 9) |
| Context Precision/Recall low | `k` (candidate pool) too small | Increase `retrieval.k` in config — proven optimal at `k=15` with this dataset |

### Deleting root-owned files created by containers

Containers run as `root` by default, so files written to mounted directories are owned by root:

```bash
# Never do: rm -rf ./storage/milvus  (permission denied)

# Instead, use a temporary Alpine container to delete:
docker run --rm \
  -v $(pwd)/storage/milvus:/target \
  alpine \
  sh -c "rm -rf /target/*"
```

---

## Quick Deployment Checklist

```
□  nvidia-smi returns both GPUs
□  Ollama running with GPU: curl http://localhost:11434/api/tags
□  Model pulled (e.g., qwen2.5:7b)
□  mkdir -p storage/etcd storage/minio storage/milvus storage/hf_cache
□  DOCKER_BUILDKIT=0 docker compose build  (no errors)
□  docker compose up -d
□  curl http://localhost:8000/health → {"status":"healthy","milvus":"connected"}
□  POST /ingest → logs show "Indexed N chunks"
□  POST /query → receives a grounded answer with sources
□  Port 8000 open on server firewall
```
