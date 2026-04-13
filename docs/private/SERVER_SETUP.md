# Server Setup Guide — MMORE RAG on Docker

Step-by-step guide for your server: **2x GTX 1080, i7, Docker-only (no sudo), Ollama already running**.

---

## What You'll End Up With

```
Server (your-server-ip)
├── Ollama          (:11434)  ← already running, serves the LLM
├── MMORE RAG API   (:8000)  ← answers questions using RAG
├── MMORE Index API (:8001)  ← (optional) upload new docs on the fly
└── Milvus Lite     (file)   ← vector DB, no server needed
```

Your frontend calls `POST http://your-server:8000/rag` and gets AI answers.

---

## Prerequisites Checklist

- [ ] Docker installed and working (no sudo needed — you have this)
- [ ] Ollama container running with a model pulled
- [ ] Your processed PDFs (`outputs/postprocess/merged/results.jsonl` from your local machine)
- [ ] NVIDIA Container Toolkit installed (for GPU access in Docker)

---

## Step 1: Verify Ollama

```bash
# Check Ollama is running and has a model
curl http://localhost:11434/api/tags

# If you need to pull a model (7B fits in one GTX 1080):
docker exec -it <ollama-container> ollama pull qwen2.5:7b

# Test it works
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hello"}]}'
```

> **Model recommendations for 2x GTX 1080 (8GB VRAM each):**
> | Model | VRAM | Quality | Speed |
> |-------|------|---------|-------|
> | `qwen2.5:3b` | ~3 GB | OK | Fast |
> | `qwen2.5:7b` | ~5 GB | Good | Medium |
> | `llama3.1:8b` | ~5 GB | Good | Medium |
> | `gemma2:9b` | ~6 GB | Better | Slower |
> | `qwen2.5:14b` | ~10 GB ⚠️ | Best | Needs 2 GPUs |

---

## Step 2: Prepare the MMORE Project

### Option A: Docker (recommended for your server)

Create a project directory on your server:

```bash
mkdir -p ~/mmore-rag && cd ~/mmore-rag

# Clone the repo
git clone https://github.com/swiss-ai/mmore.git .
```

### Transfer your processed files from local machine

On your **local Windows machine**, copy the output files to the server:

```powershell
# From your local machine
scp -r outputs/postprocess/merged/results.jsonl user@your-server:~/mmore-rag/outputs/postprocess/merged/
scp -r configs/ user@your-server:~/mmore-rag/configs/
```

Or if you want to re-process on the server, just transfer the raw PDFs and run the full pipeline there.

---

## Step 3: Create Config Files on the Server

### `configs/sk_rektor_index.yaml` — for indexing

```yaml
indexer:
  dense_model:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    is_multimodal: false
  sparse_model:
    model_name: splade
    is_multimodal: false
  db:
    uri: /app/data/sk_rektor.db
    name: sk_rektor_db
collection_name: sk_rektor_docs
documents_path: /app/outputs/postprocess/merged/results.jsonl
```

### `configs/sk_rektor_rag_api.yaml` — for the RAG API

```yaml
rag:
  llm:
    llm_name: qwen2.5:7b                          # ← change to your Ollama model
    base_url: http://host.docker.internal:11434/v1 # ← Ollama from inside Docker
    max_new_tokens: 1200
    temperature: 0.7
  retriever:
    db:
      uri: /app/data/sk_rektor.db
      name: sk_rektor_db
    hybrid_search_weight: 0.5
    k: 5
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base
  system_prompt: "Gunakan konteks berikut untuk menjawab pertanyaan. Jika konteks tidak menjawab pertanyaan, katakan tidak tahu.\n\nKonteks:\n{context}"
mode: api
mode_args:
  endpoint: '/rag'
  port: 8000
  host: '0.0.0.0'
```

> **Important URLs depending on your setup:**
> | Ollama runs as... | Use this `base_url` |
> |---|---|
> | Docker container (same network) | `http://ollama:11434/v1` |
> | Docker container (host network) | `http://localhost:11434/v1` |
> | Docker container (bridge) | `http://host.docker.internal:11434/v1` |
> | Bare metal on server | `http://localhost:11434/v1` |

---

## Step 4: Build & Run with Docker Compose

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  # ── 1. Index the documents (run once, then stop) ──
  mmore-index:
    build: .
    volumes:
      - ./configs:/app/configs
      - ./outputs:/app/outputs
      - mmore-data:/app/data
    command: >
      sh -c "
        uv pip install nltk tiktoken &&
        python -m mmore index
          --config-file /app/configs/sk_rektor_index.yaml
          --documents-path /app/outputs/postprocess/merged/results.jsonl
      "
    profiles: ["setup"]   # Only runs when you explicitly call it

  # ── 2. RAG API (runs forever, serves your frontend) ──
  mmore-rag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./configs:/app/configs
      - mmore-data:/app/data
    environment:
      - OPENAI_API_KEY=dummy
    command: >
      sh -c "
        uv pip install nltk tiktoken &&
        python -m mmore rag --config-file /app/configs/sk_rektor_rag_api.yaml
      "
    extra_hosts:
      - "host.docker.internal:host-gateway"  # So container can reach Ollama
    restart: unless-stopped

  # ── 3. Index API (optional — for adding docs on the fly) ──
  mmore-index-api:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./configs:/app/configs
      - mmore-data:/app/data
    command: >
      python -m mmore index-api
        --config-file /app/configs/sk_rektor_index.yaml
        --host 0.0.0.0 --port 8001
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    profiles: ["full"]   # Only if you want document management

volumes:
  mmore-data:   # Persists the Milvus Lite .db file
```

### Run it

```bash
# ── ONE-TIME SETUP ──

# 1. Build the image
docker compose build

# 2. Index your documents (creates the vector DB)
docker compose --profile setup run --rm mmore-index

# ── START THE API ──

# 3. Start the RAG API (runs in background)
docker compose up -d mmore-rag

# Check it's running
docker compose logs -f mmore-rag

# Test it
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": "Apa ketentuan biaya pendidikan di UI?",
      "collection_name": "sk_rektor_docs"
    }
  }'
```

---

## Step 5: Alternative — Run Without Docker (bare metal with venv)

If you can install Python directly on the server (without sudo):

```bash
# 1. Clone & setup
git clone https://github.com/swiss-ai/mmore.git
cd mmore

# 2. Install uv (no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create venv and install
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all,cpu]"
uv pip install nltk tiktoken

# For GPU support (your 2x GTX 1080):
# uv pip install -e ".[all,cu126]"

# 4. Copy your processed files from local machine
# (or re-run process + postprocess on the server)

# 5. Index
python -m mmore index \
  --config-file configs/sk_rektor_index.yaml \
  --documents-path outputs/postprocess/merged/results.jsonl

# 6. Start RAG API
export OPENAI_API_KEY=dummy
python -m mmore rag --config-file configs/sk_rektor_rag_api.yaml
# → API running at http://0.0.0.0:8000/rag
```

> **Tip**: Use `tmux` or `screen` to keep the process running after you disconnect SSH:
> ```bash
> tmux new -s mmore-rag
> python -m mmore rag --config-file configs/sk_rektor_rag_api.yaml
> # Press Ctrl+B then D to detach
> # Reconnect later with: tmux attach -t mmore-rag
> ```

---

## API Reference (what your frontend calls)

### `POST /rag` — Ask a question

```bash
curl -X POST http://your-server:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": "Your question here",
      "collection_name": "sk_rektor_docs"
    }
  }'
```

**Response:**
```json
{
  "input": "Your question here",
  "context": "[1] Relevant text from SK Rektor...\n\n[2] Another relevant chunk...",
  "answer": "The AI's answer based on the context..."
}
```

### `GET /health` — Check if API is alive

```bash
curl http://your-server:8000/health
# → {"status": "healthy"}
```

### Swagger Docs

Visit `http://your-server:8000/docs` in your browser for interactive API documentation.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY not found` | Set `export OPENAI_API_KEY=dummy` (Ollama doesn't need a real key) |
| `Connection refused` to Ollama | Check the `base_url` — use `host.docker.internal` from Docker, `localhost` from bare metal |
| Milvus errors | Make sure the `.db` file path is writable and consistent between index and rag configs |
| Out of memory | Use a smaller model (`qwen2.5:3b`) or reduce `k` (number of retrieved chunks) |
| Slow first response | First query downloads embedding models (~80 MB). Subsequent queries are fast |
| CORS errors from frontend | See CORS middleware instructions in `GUIDE_MMORE_PIPELINE.md` Section 8 |

---

## Quick Checklist

```
□ Ollama running with model pulled
□ Processed JSONL files transferred to server
□ Config files created (index.yaml + rag_api.yaml)
□ Indexed documents into Milvus Lite (one-time)
□ RAG API started and responding to /health
□ Frontend can reach http://your-server:8000/rag
```
