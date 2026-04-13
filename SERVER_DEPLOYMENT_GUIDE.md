# 🚀 Master Deployment Guide: Research Server (GTX 1080 Dual GPU)

This guide provides the complete, end-to-end setup for your upgraded RAG system on the `riset-01` server.

---

## 1. Prerequisites (Host Level)
Ensure these are active on your server before proceeding:
- **Ollama**: Running on port `11434`.
- **NVIDIA Driver**: Installed and working (`nvidia-smi` should return results).
- **Docker**: Versions >= 20.10.8.

---

## 2. Clone & Prepare
Run these commands in your user home directory:

```bash
# 1. Clone the repository
git clone https://github.com/aryakdaniswara/my-rag.git
cd my-rag

# 2. Pull latest fixes (if already cloned)
git pull origin main

# 3. Create required directories for persistence
mkdir -p storage/etcd storage/minio storage/milvus storage/hf_cache
```

---

## 3. Initialize the Stack
We use a multi-container setup. The first run will download the base images and initialized the databases.

```bash
# 1. Build the RAG API image (Fixing BuildKit for your server version)
DOCKER_BUILDKIT=0 docker compose build

# 2. Start the system (Milvus Standalone + RAG API)
docker compose up -d

# 3. Monitor initialization
docker compose logs -f rag-api
```

---

## 4. Ingest Your Research Data
Once the containers are healthy, trigger the ingestion of your university regulations:

```bash
# Using curl to trigger the background ingestion task
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data"}'
```
*Note: This will process all PDFs in your `./data` folder using Docling and Harrier.*

---

## 5. First Query Test
Verify the hybrid retrieval and reranking with a sample query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa itu mekanisme penelaahan usulan pembukaan program studi?"}'
```

---

## 🛠️ Server Troubleshooting & Features

### GPU Allocation
- **GPU 0**: Handled by Harrier (Dense) & OpenSearch (Sparse).
- **GPU 1**: Dedicated to Jina-v3 (Reranker).
*This is automatically managed by the `docker-compose.yml` environment.*

### Stability Flags
We have already baked these into the `docker-compose.yml` to prevent the crashes you previously experienced:
- `pids_limit: -1`
- `security_opt: [seccomp:unconfined]`
- `network: bridge` (connecting to Ollama via `host.docker.internal`)

### Persistence
- Your vectors are stored in `./storage/milvus`.
- Your model weights are cached in `./storage/hf_cache` (so they don't redownload).

---

## API Swagger UI
Once deployed, you can access the full API documentation at:
`http://<SERVER_IP>:8000/docs`
