# MMORE Pipeline Guide — SK Rektor PSA PDFs

A step-by-step guide to replicate the MMORE pipeline setup on Windows for processing PDF documents into structured markdown, and optionally extending to embedding, indexing, and RAG.

---

## Table of Contents

1. [What is MMORE?](#1-what-is-mmore)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Stage 1: Process (PDF → Structured Markdown)](#4-stage-1-process-pdf--structured-markdown)
5. [Stage 2: Post-Process (Chunking)](#5-stage-2-post-process-chunking)
6. [Stage 3: Index (Embedding → Vector DB)](#6-stage-3-index-embedding--vector-db)
7. [Stage 4: RAG (Chatbot)](#7-stage-4-rag-chatbot)
8. [Server Deployment & API Architecture](#8-server-deployment--api-architecture)
9. [Output Format Reference](#9-output-format-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What is MMORE?

**MMORE (Massive Multimodal Open RAG & Extraction)** is an **orchestration/integration framework**, not a single monolithic tool. It consolidates and wraps multiple existing open-source libraries into a unified pipeline:

| Pipeline Stage | Underlying Libraries | What They Do |
|---|---|---|
| **PDF Processing (default)** | [marker-pdf](https://github.com/VikParuchuri/marker) + [surya-ocr](https://github.com/VikParuchuri/surya) | OCR, layout detection, structured text extraction |
| **PDF Processing (fast)** | [PyMuPDF](https://pymupdf.readthedocs.io/) | Direct text/image extraction (no OCR) |
| **DOCX Processing** | [python-docx](https://python-docx.readthedocs.io/) | Word document parsing |
| **Media Processing** | [moviepy](https://zulko.github.io/moviepy/) + [OpenAI Whisper](https://github.com/openai/whisper) | Video frame extraction + audio transcription |
| **HTML Processing** | [markdownify](https://pypi.org/project/markdownify/) + [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | HTML → Markdown conversion |
| **Chunking** | [chonkie](https://github.com/bhavnicksm/chonkie) + [nltk](https://www.nltk.org/) | Sentence/token-based text splitting |
| **Indexing** | [Milvus](https://milvus.io/) + [sentence-transformers](https://www.sbert.net/) | Dense+sparse vector embeddings and storage |
| **RAG** | [LangChain](https://www.langchain.com/) | LLM integration with retrieval |
| **Distributed** | [Dask](https://distributed.dask.org/) | Multi-node/GPU parallelism |

**In short**: MMORE is a consolidation/glue framework. It provides:
- A **unified data format** (`MultimodalSample`) across all file types
- A **dispatcher** that routes files to the right processor
- **Config-driven YAML pipelines** so you don't write code
- **CLI commands** (`mmore process`, `mmore index`, `mmore rag`) for each stage

It does NOT implement its own OCR, embedding model, or LLM — it wraps the best existing tools.

---

## 2. Prerequisites

- **OS**: Windows 10/11 (tested), Linux, or macOS
- **Python**: 3.10 – 3.12 (we used 3.12.3)
- **uv**: Astral's fast package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Hardware used**: Intel i5-10300H + NVIDIA GTX 1660 MaxQ (6GB VRAM), 16GB RAM
  - Fast mode (PyMuPDF) runs on CPU only, no GPU needed
  - Default mode (marker-pdf) benefits from GPU

### Install uv (if not installed)

```powershell
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 3. Environment Setup

```powershell
# Navigate to the MMORE repository
cd C:\Users\aryan\Documents\kuliah\skripsi\mmore

# Create a virtual environment with Python 3.12
uv venv .venv --python 3.12

# Activate the virtual environment
.venv\Scripts\activate

# Install process-only dependencies (CPU mode)
uv pip install -e ".[process,cpu]"

# Install additional required dependencies (not in the process extra)
uv pip install nltk tiktoken
```

> **Note on install extras**:
> - `process,cpu` — PDF/doc processing with CPU-only PyTorch (~590 MB torch download)
> - `all,cpu` — Everything (process + index + rag + api) with CPU PyTorch
> - `all,cu126` — Everything with CUDA 12.6 GPU support

---

## 4. Stage 1: Process (PDF → Structured Markdown)

### Config File

Create `configs/sk_rektor_process.yaml`:

```yaml
data_path: C:/Users/aryan/Documents/kuliah/skripsi/mmore/SK_Rektor_PSA/
google_drive_ids: []
dispatcher_config:
  output_path: C:/Users/aryan/Documents/kuliah/skripsi/mmore/outputs/process/
  use_fast_processors: true          # true = PyMuPDF (fast), false = marker-pdf (OCR)
  distributed: false
  dashboard_backend_url: null
  extract_images: true
  scheduler_file: null
  process_batch_sizes:
    - PDFProcessor: 4                # small batch size for limited hardware
  processor_config:
    PDFProcessor:
      - PDFTEXT_CPU_WORKERS: 1       # limit CPU workers
      - DETECTOR_BATCH_SIZE: 1
      - DETECTOR_POSTPROCESSING_CPU_WORKERS: 1
      - RECOGNITION_BATCH_SIZE: 1
      - OCR_PARALLEL_WORKERS: 1
      - TEXIFY_BATCH_SIZE: 1
      - LAYOUT_BATCH_SIZE: 1
      - ORDER_BATCH_SIZE: 1
      - TABLE_REC_BATCH_SIZE: 1
```

### Run

```powershell
.venv\Scripts\activate
python -m mmore process --config-file configs/sk_rektor_process.yaml
```

### Output

```
outputs/process/
├── processors/
│   └── PDFProcessor/
│       └── results.jsonl          # Per-processor results
├── merged/
│   └── merged_results.jsonl       # All results merged (24 entries for 24 PDFs)
└── images/                        # Extracted images from PDFs
```

**Performance**: 24 PDFs (~150 MB total) processed in ~316 seconds with fast mode.

### Fast Mode vs Default Mode

| | Fast Mode (`true`) | Default Mode (`false`) |
|---|---|---|
| **Tool** | PyMuPDF | marker-pdf + surya-ocr |
| **Speed** | Very fast (~5 min for 24 PDFs) | Slower (can be 10-30x slower) |
| **OCR** | No OCR (text extraction only) | Full OCR support |
| **Quality** | Good for native/digital PDFs | Better for scanned/image PDFs |
| **GPU** | Not needed | Recommended (speeds up OCR) |

> **Tip**: If your PDFs are scanned documents or contain mostly images, switch to `use_fast_processors: false` and use GPU if available.

---

## 5. Stage 2: Post-Process (Chunking)

### Config File

Create `configs/sk_rektor_postprocess.yaml`:

```yaml
pp_modules:
  - type: chunker
    args:
      chunking_strategy: sentence    # Splits text into sentence-based chunks

output:
  output_path: C:/Users/aryan/Documents/kuliah/skripsi/mmore/outputs/postprocess/merged/results.jsonl
  save_each_step: True
```

### Run

```powershell
.venv\Scripts\activate
python -m mmore postprocess --config-file configs/sk_rektor_postprocess.yaml --input-data outputs/process/merged/merged_results.jsonl
```

### Output

```
outputs/postprocess/
└── merged/
    └── results.jsonl               # 625 chunks from 24 PDFs
```

**Result**: 24 documents → 625 sentence-based chunks ready for embedding/indexing.

---

## 6. Stage 3: Index (Embedding → Vector DB)

> **This stage was NOT run in our setup.** Below are the instructions to extend.

### Additional Installation

```powershell
.venv\Scripts\activate
uv pip install -e ".[index,cpu]"   # or ".[index,cu126]" for GPU
```

### Config File

Create `configs/sk_rektor_index.yaml`:

```yaml
indexer:
  dense_model:
    model_name: sentence-transformers/all-MiniLM-L6-v2   # Lightweight embedding model
    is_multimodal: false
  sparse_model:
    model_name: splade                                    # Sparse retrieval model
    is_multimodal: false
  db:
    uri: ./sk_rektor.db                                   # Local Milvus Lite DB file
    name: sk_rektor_db
collection_name: sk_rektor_docs
documents_path: 'outputs/postprocess/merged/results.jsonl'
```

### Run

```powershell
python -m mmore index --config-file configs/sk_rektor_index.yaml --documents-path outputs/postprocess/merged/results.jsonl
```

### What Happens

1. Each chunk from `results.jsonl` is embedded using:
   - **Dense embeddings**: `all-MiniLM-L6-v2` (384-dim sentence vectors)
   - **Sparse embeddings**: SPLADE (for keyword-based retrieval)
2. Embeddings are stored in a **Milvus Lite** local database file (`sk_rektor.db`)
3. This enables **hybrid search** (dense + sparse) for better retrieval accuracy

### Alternative Embedding Models

| Model | Dims | Size | Speed | Quality |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 80 MB | Fast | Good |
| `all-mpnet-base-v2` | 768 | 420 MB | Medium | Better |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3 GB | Slow | Best |
| `BAAI/bge-m3` | 1024 | 2.2 GB | Slow | Best (multilingual) |

> **For Indonesian documents like SK_Rektor_PSA**, consider using `BAAI/bge-m3` which supports 100+ languages including Indonesian.

---

## 7. Stage 4: RAG (Chatbot)

> **This stage was NOT run in our setup.** Below are the instructions to extend.

### Additional Installation

```powershell
.venv\Scripts\activate
uv pip install -e ".[rag,cpu]"     # Includes index dependencies + LangChain
```

### Config File — Local Mode

Create `configs/sk_rektor_rag.yaml` (runs locally with a HuggingFace model):

```yaml
rag:
  llm:
    llm_name: mistralai/Mistral-7B-Instruct-v0.3   # Or any HF model
    max_new_tokens: 1200
  retriever:
    db:
      uri: ./sk_rektor.db
      name: 'sk_rektor_db'
    hybrid_search_weight: 0.5
    k: 5
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base
  system_prompt: "Use the following context to answer the questions.\n\nContext:\n{context}"
mode: local
mode_args:
  input_file: configs/queries.jsonl        # Your questions
  output_file: outputs/rag/output.json     # Answers
```

### Config File — API Mode (with OpenAI)

Create `configs/sk_rektor_rag_api.yaml`:

```yaml
rag:
  llm:
    llm_name: gpt-4o-mini                 # OpenAI model
    max_new_tokens: 1200
  retriever:
    db:
      uri: ./sk_rektor.db
      name: 'sk_rektor_db'
    hybrid_search_weight: 0.5
    k: 5
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base
  system_prompt: "Use the following context to answer the questions.\n\nContext:\n{context}"
mode: api
mode_args:
  host: localhost
  port: 8000
```

> **Note**: For OpenAI, set `OPENAI_API_KEY` environment variable first.

### Run

```powershell
# Local mode (batch questions)
python -m mmore rag --config-file configs/sk_rektor_rag.yaml

# API mode (starts a server at localhost:8000)
python -m mmore rag --config-file configs/sk_rektor_rag_api.yaml

# Interactive CLI chat mode
python -m mmore ragcli --config-file configs/sk_rektor_rag.yaml
```

### Query the API

```powershell
curl -X POST http://localhost:8000/rag/invoke `
  -H "Content-Type: application/json" `
  -d '{"input": {"input": "Apa itu biaya pendidikan di UI?", "collection_name": "sk_rektor_docs"}}'
```

### Alternative LLM Options

| Option | Pros | Cons |
|---|---|---|
| **OpenAI API** (`gpt-4o-mini`) | Easy, high quality | Costs money, needs internet |
| **Local Ollama** | Free, private | Needs good GPU (≥8GB VRAM) |
| **HuggingFace model** | Free, flexible | Large downloads, needs GPU |
| **vLLM server** | Fast local inference | Complex setup |

> **For your hardware (GTX 1660 MaxQ, 6GB VRAM)**: Use OpenAI API or a small local model like `Qwen/Qwen2.5-3B-Instruct` via Ollama. A 7B model may not fit in 6GB VRAM.

---

## 8. Server Deployment & API Architecture

This section covers how to deploy MMORE on your server (2x GTX 1080, Docker-only) as an API that a frontend can call.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR SERVER (Docker)                     │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│  │  Ollama   │   │  MMORE RAG   │   │  MMORE Index API    │ │
│  │  :11434   │   │  API :8000   │   │  (optional) :8001   │ │
│  │  (LLM)    │◄──│  /rag        │   │  /v1/files          │ │
│  └──────────┘   │  /health     │   │  (CRUD + auto-index)│ │
│                  │              │   └─────────────────────┘ │
│                  │   ┌────────┐ │                           │
│                  │   │Milvus  │ │                           │
│                  │   │Lite .db│ │                           │
│                  │   └────────┘ │                           │
│                  └──────────────┘                           │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
                   ┌────┴─────┐
                   │ Frontend │
                   │ (Vue/React)│
                   └──────────┘
```

### MMORE Exposes 3 Separate API Servers

You can run one or multiple depending on your needs:

#### API 1: RAG API (the main one for your chatbot)

**What it does**: Accepts a question, retrieves relevant chunks from Milvus, sends them + the question to Ollama, returns the answer.

**Start command**:
```bash
python -m mmore rag --config-file configs/sk_rektor_rag_api.yaml
```

**Endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rag` | Ask a question, get RAG answer |
| `GET` | `/health` | Health check |

**Request** (`POST /rag`):
```json
{
  "input": {
    "input": "Apa ketentuan biaya pendidikan di UI?",
    "collection_name": "sk_rektor_docs"
  }
}
```

**Response**:
```json
{
  "input": "Apa ketentuan biaya pendidikan di UI?",
  "context": "[1] Relevant chunk from PDF...\n\n[2] Another relevant chunk...",
  "answer": "Berdasarkan Peraturan Rektor No. 2 Tahun 2025, biaya pendidikan di UI..."
}
```

**Config file** (`configs/sk_rektor_rag_api.yaml`):
```yaml
rag:
  llm:
    llm_name: qwen2.5:7b                        # Your Ollama model name
    base_url: http://localhost:11434/v1           # Ollama's OpenAI-compatible API
    max_new_tokens: 1200
    temperature: 0.7
  retriever:
    db:
      uri: ./sk_rektor.db                        # Milvus Lite (just a file)
      name: sk_rektor_db
    hybrid_search_weight: 0.5
    k: 5                                         # Return top 5 chunks
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base  # Re-ranks results for quality
  system_prompt: "Gunakan konteks berikut untuk menjawab pertanyaan. Jika tidak ada konteks yang menjawab, katakan tidak tahu.\n\nKonteks:\n{context}"
mode: api
mode_args:
  endpoint: '/rag'
  port: 8000
  host: '0.0.0.0'                               # Accessible from outside
```

> **How Ollama works with MMORE**: In MMORE's `llm.py`, when `base_url` is set and the model name doesn't match known providers (OpenAI, Anthropic, etc.), it uses LangChain's `ChatOpenAI` with `base_url`. Since Ollama exposes an OpenAI-compatible API at `/v1`, this works seamlessly. You need to set `OPENAI_API_KEY=dummy` as an environment variable (Ollama ignores it but LangChain requires it).

---

#### API 2: Index API (optional — for adding/managing documents on the fly)

**What it does**: Full CRUD for documents. Upload a PDF → it auto-processes → auto-indexes into Milvus. No need to run the process/postprocess steps manually.

**Start command**:
```bash
python -m mmore index-api --config-file configs/sk_rektor_index.yaml --host 0.0.0.0 --port 8001
```

**Endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/files` | Upload & index a single file |
| `POST` | `/v1/files/bulk` | Upload & index multiple files |
| `PUT` | `/v1/files/{fileId}` | Replace & re-index a file |
| `DELETE` | `/v1/files/{fileId}` | Delete a file + its vectors |
| `GET` | `/v1/files/{fileId}` | Download a file |

**Example — Upload a new PDF**:
```bash
curl -X POST http://your-server:8001/v1/files \
  -F "fileId=pr_2025_no_16" \
  -F "file=@/path/to/new_document.pdf"
```

**Swagger UI**: Visit `http://your-server:8001/docs` for interactive API docs.

---

#### API 3: Retriever API (optional — search-only, no LLM)

**What it does**: Search the vector DB without calling an LLM. Useful if your frontend wants to show "source documents" separately from the AI answer.

**Start command**:
```bash
python -m mmore retrieve --config-file configs/sk_rektor_index.yaml --host 0.0.0.0 --port 8002
```

---

### Complete Server Deployment Steps

#### Step 1: Install everything on the server

```bash
# Clone the repo
git clone https://github.com/swiss-ai/mmore.git
cd mmore

# Create venv
uv venv .venv --python 3.12
source .venv/bin/activate     # Linux server

# Install ALL modules (process + index + rag + api)
uv pip install -e ".[all,cpu]"
uv pip install nltk tiktoken

# If your server has CUDA GPUs and you want GPU acceleration:
# uv pip install -e ".[all,cu126]"
```

#### Step 2: Process your PDFs (one-time)

```bash
# Copy your PDFs to the server, then:
python -m mmore process --config-file configs/sk_rektor_process.yaml
python -m mmore postprocess --config-file configs/sk_rektor_postprocess.yaml \
  --input-data outputs/process/merged/merged_results.jsonl
```

#### Step 3: Index into Milvus (one-time)

```bash
python -m mmore index --config-file configs/sk_rektor_index.yaml \
  --documents-path outputs/postprocess/merged/results.jsonl
```

This creates `sk_rektor.db` (Milvus Lite file) with all 625 chunks embedded.

#### Step 4: Start the RAG API

```bash
# Set dummy API key (Ollama doesn't need one, but LangChain requires it)
export OPENAI_API_KEY=dummy

# Make sure Ollama is running with your model
# ollama run qwen2.5:7b

# Start MMORE RAG API
python -m mmore rag --config-file configs/sk_rektor_rag_api.yaml
```

The API is now live at `http://your-server:8000/rag`.

#### Step 5 (optional): Start the Index API for document management

```bash
# In another terminal
python -m mmore index-api --config-file configs/sk_rektor_index.yaml --host 0.0.0.0 --port 8001
```

### Docker Deployment (no sudo needed)

If you prefer Docker, create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  mmore-rag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./configs:/app/configs
      - ./outputs:/app/outputs
      - ./sk_rektor.db:/app/sk_rektor.db
    environment:
      - OPENAI_API_KEY=dummy
    command: python -m mmore rag --config-file /app/configs/sk_rektor_rag_api.yaml
    depends_on:
      - ollama
    network_mode: host     # So it can reach Ollama on localhost

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Connecting from a Frontend

Your frontend (Vue.js, React, etc.) calls the RAG API:

```javascript
// Frontend JavaScript example
async function askQuestion(question) {
  const response = await fetch('http://your-server:8000/rag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: {
        input: question,
        collection_name: 'sk_rektor_docs'
      }
    })
  });

  const data = await response.json();
  // data.answer  -> The AI's response
  // data.context -> The source chunks used
  // data.input   -> The original question
  return data;
}
```

> **CORS**: If your frontend is on a different domain, you may need to add CORS middleware to MMORE's FastAPI app. Add this in `src/mmore/run_rag.py`:
> ```python
> from fastapi.middleware.cors import CORSMiddleware
> app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
> ```

---

## 9. Output Format Reference

Each entry in the JSONL files follows the `MultimodalSample` schema:

```json
{
  "text": "# Title\n\nExtracted markdown text content...\n\n## Section 1\n...",
  "modalities": [
    {
      "type": "image",
      "value": "path/to/extracted_image.png"
    }
  ],
  "metadata": {
    "file_path": "C:/Users/.../SK_Rektor_PSA/document.pdf",
    "document_type": "pdf"
  }
}
```

---

## 10. Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: nltk` | `uv pip install nltk` |
| `ModuleNotFoundError: tiktoken` | `uv pip install tiktoken` |
| `torch` download slow | Use `uv` (faster than pip), or pre-download the wheel |
| Out of memory (default mode) | Use fast mode (`use_fast_processors: true`) or reduce batch sizes |
| Milvus errors on Windows | Use `pymilvus[milvus-lite]` — it bundles a local DB, no server needed |
| `marker-pdf` model download | First run downloads ~1-2 GB of models; be patient |
| PDFs with scanned images | Switch to `use_fast_processors: false` for OCR capability |

---

## Quick Reference — Full Pipeline Commands

```powershell
# 0. Setup
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -e ".[all,cpu]"
uv pip install nltk tiktoken

# 1. Process PDFs
python -m mmore process --config-file configs/sk_rektor_process.yaml

# 2. Post-process (chunk)
python -m mmore postprocess --config-file configs/sk_rektor_postprocess.yaml --input-data outputs/process/merged/merged_results.jsonl

# 3. Index into vector DB
python -m mmore index --config-file configs/sk_rektor_index.yaml --documents-path outputs/postprocess/merged/results.jsonl

# 4. Run RAG chatbot
python -m mmore rag --config-file configs/sk_rektor_rag.yaml
```
