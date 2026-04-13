# Deploying MMORE RAG with Docker & Ollama

This document records every problem encountered while setting up the MMORE RAG pipeline on a research server (`riset-01`, Ubuntu, Docker 20.10.8, 2× NVIDIA GTX 1080), how each was diagnosed, and how it was fixed.

---

## 0. Project Vision: The Universitas Indonesia One-Stop Help Desk

This deployment is part of a larger, hybrid infrastructure designed to provide automated support for Universitas Indonesia (UI) students and staff. The goal is to build a **One-Stop Help Desk Chatbot** where users can ask any question about the university's regulations and services.

### Hybrid Architecture (The Stack)
- **Frontend (Vue.js)**: The interactive web interface for the chatbot and ticket tracking system.
- **Core Backend (Go/Golang)**: Handles authentication, persistent storage, and the overarching **Ticketing System**.
- **AI RAG System (Python/MMORE)**: The containerized intelligence engine (which this document covers). It is responsible for factual retrieval from UI regulations (e.g., SK Rektor).

### Automated Fallback Logic
The RAG Chatbot is designed to triage requests:
1. **Direct Resolution**: If the RAG engine successfully finds an answer in the vector database, the response is served immediately to the user.
2. **Ticketing Fallback**: If the RAG engine is unsure, lacks context, or cannot find the information, the system triggers a fallback flow. The user is prompted to **Create a Ticket**, which is then routed to and handled by the Go-based backend for human intervention.

### Deployment Context
Docker is used specifically to isolate this Python-based AI environment (with its heavy ML dependencies and GPU requirements) from the lightweight, high-concurrency Go backend, while allowing both services to communicate seamlessly over the local network.

---

## Table of Contents

0. [Project Vision: The Universitas Indonesia One-Stop Help Desk](#0-project-vision-the-universitas-indonesia-one-stop-help-desk)
1. [Ollama Setup & GPU Access](#1-ollama-setup--gpu-access)
2. [MMORE Code Bug Fixes](#2-mmore-code-bug-fixes)
3. [Dependency Hell (pip vs uv, transformers version)](#3-dependency-hell)
4. [Docker Build Failures](#4-docker-build-failures)
5. [Docker Runtime Issues](#5-docker-runtime-issues)
6. [RAG API Response Issues](#6-rag-api-response-issues)
7. [Network & Firewall](#7-network--firewall)
8. [Final Working Configuration](#8-final-working-configuration)
9. [API Usage](#9-api-usage)

---

## 1. Ollama Setup & GPU Access

MMORE's RAG pipeline uses an LLM for the generation step. We used [Ollama](https://ollama.ai) to serve models locally via an OpenAI-compatible API.

### Problem: Ollama running on CPU only
Ollama was originally started without GPU flags, so inference on `qwen2.5:latest` took **minutes** per query.

### Fix: Recreate Ollama container with GPU access
```bash
docker rm -f ollama
docker run -d --name ollama \
  --gpus all --privileged \
  -v ollama_data:/root/.ollama \
  -p 11434:11434 \
  --restart unless-stopped \
  ollama/ollama
```
Models (`qwen2.5`, `gemma2`, etc.) were preserved in the `ollama_data` Docker volume, so nothing was lost. After this change, inference dropped from minutes to **seconds**.

### How MMORE connects to Ollama
MMORE uses LangChain's `ChatOpenAI` class, which speaks the OpenAI chat completions protocol. Ollama exposes an OpenAI-compatible endpoint at `/v1`. The connection is configured in the YAML config:

```yaml
rag:
  llm:
    llm_name: qwen2.5:latest
    base_url: http://localhost:11434/v1   # Ollama's OpenAI-compatible endpoint
```

A dummy `OPENAI_API_KEY` environment variable must be set (Ollama doesn't check it, but LangChain requires it):
```bash
-e OPENAI_API_KEY=dummy
```


## 1.1. Model Persistence & Volume Consolidation

### **Problem: Model Fragmentation**
Akibat banyaknya eksperimen (penggunaan MMORE, instalasi manual, dsb.), model AI tersebar di beberapa Docker volume seperti `ollama_data`, `ollama_storage`, dan `mmore_ollama_data`. 

**Dampak Negatif:**
* **Inefisiensi Ruang Disk:** Penggunaan storage di server `riset-01` menjadi boros karena redundansi file.
* **Kebingungan Deployment:** Sulit menentukan volume mana yang berisi versi model terbaru saat akan menjalankan kontainer.

---

### **Diagnosis: Inspecting Volume Contents**
Untuk memeriksa isi volume tanpa memerlukan akses `sudo` langsung ke direktori root Docker, kita menggunakan teknik **bridge container** (kontainer sementara menggunakan image Alpine yang ringan):

```bash
docker run --rm -v <nama_volume>:/data alpine ls -R /data/models/manifests
```

**Temuan Lapangan:**
* **`ollama_storage`**: Memiliki manifest terlengkap (termasuk model terbaru seperti `qwen3.5:9b` dan `gemma3`).
* **`ollama_data`**: Hanya berisi model versi lama.

---

### **Fix: The "Bridge Sync" Method**
Konsolidasi model dilakukan ke satu volume utama (`ollama_storage`) menggunakan teknik *non-destructive copy*. Metode ini menggabungkan isi dua volume tanpa menghapus data yang sudah ada.

**Eksekusi Perintah:**
```bash
# Sinkronisasi model dari volume lama ke volume utama
docker run --rm \
  -v ollama_data:/from \
  -v ollama_storage:/to \
  alpine sh -c "cp -rn /from/* /to/"
```

> [!TIP]
> **Catatan Teknis:** Penggunaan flag `-n` (*no-clobber*) sangat krusial untuk memastikan file yang sudah ada di target tidak tertimpa atau rusak selama proses migrasi.

---

### **Finalized Volume Strategy**
Kami menetapkan **`ollama_storage`** sebagai volume permanen tunggal. Pemilihan nama ini didasarkan pada pemisahan semantik yang jelas:
* **Aplikasi:** Logika operasional Ollama.
* **Storage:** Aset model AI (blob) yang berukuran besar (GigaBytes).

#### **Updated Run Command (Production Ready)**
Gunakan perintah berikut untuk menjalankan Ollama dengan konfigurasi volume yang sudah terkonsolidasi:

```bash
docker rm -f ollama
docker run -d --name ollama \
  --gpus all --privileged \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  --restart unless-stopped \
  ollama/ollama
```

---

## 2. MMORE Code Bug Fixes

Three bugs in the MMORE source code prevented the RAG pipeline from running. All three were runtime errors.

### A. Sparse Model — `coo_array` missing `indices` attribute

**File:** `src/mmore/rag/model/sparse/splade.py`

**Error:**
```
AttributeError: coo_array has no attribute 'indices'
```

**Root Cause:** Newer versions of `scipy` return `coo_array` from sparse operations, but the code expected a `csr_matrix` which has `.indices` and `.data` attributes. `coo_array` uses `.col` and `.data` instead.

**Fix:** Convert the sparse output to CSR format before accessing indices:
```python
# Before
sparse_output = ...  # returns coo_array
indices = sparse_output.indices

# After
sparse_output = sparse_output.tocsr()  # convert to csr_matrix
indices = sparse_output.indices
```

---

### B. Indexer — `dict` has no `reshape` attribute

**File:** `src/mmore/index/indexer.py`

**Error:**
```
AttributeError: 'dict' object has no attribute 'reshape'
```

**Root Cause:** The sparse model returned a dictionary containing sparse vector data, but the indexer code called `.reshape()` on it assuming it was a numpy array.

**Fix:** Extract the actual sparse data from the dictionary before reshaping:
```python
# Before
sparse_embed = model.encode(text)
sparse_embed = sparse_embed.reshape(...)

# After
sparse_embed = model.encode(text)
if isinstance(sparse_embed, dict):
    sparse_embed = sparse_embed  # already in the correct dict format for Milvus
```

---

### C. Retriever — Hardcoded `.to("cuda")`

**File:** `src/mmore/rag/retriever.py`

**Error:**
```
RuntimeError: No CUDA GPUs are available
```
(This occurred when running MMORE via Docker without GPU passthrough, or on a CPU-only setup.)

**Root Cause:** The reranker model loading had a hardcoded `.to("cuda")` call, which crashes on CPU-only environments.

**Fix:** Auto-detect the available device:
```python
# Before
self.reranker_model = ...
self.reranker_model.to("cuda")

# After
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
self.reranker_model.to(device)
```

---

## 3. Dependency Hell

### A. Missing modules due to incomplete `pip install`

**Problem:** After installing with `pip install -e ".[rag]"`, several modules were missing at runtime:
- `nltk` — required for sentence tokenization in the chunker
- `tiktoken` — required for token counting
- `uvicorn` / `fastapi` — required to serve the API

**Root Cause:** These packages are used by the codebase but not listed in the `[rag]` extras of `pyproject.toml`.

**Fix:** Install them explicitly:
```bash
pip install nltk tiktoken uvicorn fastapi
```

In the Dockerfile these are included in the `RUN pip install` line.

---

### B. `transformers` version incompatibility (BertTokenizer crash)

**Error:**
```
AttributeError: BertTokenizer has no attribute batch_encode_plus. Did you mean: '_encode_plus'?
```

**Root Cause:** `pip install` resolved `transformers==5.2.0` (latest at the time). Version 5.x introduced breaking API changes to the tokenizer classes — `batch_encode_plus` was renamed/removed.

**Diagnosis:** We checked the installed version inside the container:
```bash
docker exec mmore-rag pip show transformers | grep Version
# Output: Version: 5.2.0
```

**Fix:** Pin transformers to a known-compatible 4.x version:
```bash
pip install "transformers==4.48.0"
```

> **Note:** We initially tried `transformers==4.44.0` but encountered import errors with the dense model. Version `4.48.0` was the sweet spot — new enough for the dense encoder but old enough to retain `batch_encode_plus`.

---

### C. Dense model import error with wrong transformers version

**Error (with transformers==4.44.0):**
```
ImportError: cannot import name 'DenseModel' from 'mmore.rag.model.dense'
```

The import chain was:
```
cli.py → run_rag.py → pipeline.py → retriever.py → indexer.py → model/__init__.py → dense/__init__.py → base.py
```

**Root Cause:** `base.py` in the dense model module used features from transformers that weren't available in 4.44.0.

**Fix:** Bumped to `transformers==4.48.0` which resolved both the BertTokenizer issue and the dense model import.

---

## 4. Docker Build Failures

The research server (`riset-01`) runs Docker 20.10.8 with strict security profiles. This caused three distinct build failures.

### A. APT Post-Invoke Script Error

**Error:** APT hooks inside `nvidia/cuda` or `ubuntu` base images were blocked by the server's seccomp profile during `docker build`.

**Fix:** Use `python:3.12-slim` as the base image instead. No `apt-get` calls needed.

### B. `pip` Thread Creation Error

**Error:**
```
RuntimeError: can't start new thread
```

**Root Cause:** Docker's default seccomp profile blocks the `clone()` syscall during builds. pip's `rich` library spawns threads for the progress bar, which triggers this.

**Why it only fails during build:** `docker build` does NOT support `--security-opt` on Docker 20.10.8 (that feature was added in Docker 23+). So we can't relax seccomp during build.

**Fix:**
```dockerfile
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1
```

### C. `uv` Tokio Runtime Panic

**Error:**
```
Tokio executor failed: PermissionDenied
```

**Root Cause:** `uv` (Rust-based pip replacement) uses Tokio async runtime, which requires `io_uring` / `epoll_create` syscalls. The server's seccomp profile blocks these during Docker build.

**Fix:** Abandoned `uv` inside Docker; reverted to plain `pip`.

---

## 5. Docker Runtime Issues

### OpenBLAS Thread Creation Crash

**Error (when running temporary containers without security flags):**
```
OpenBLAS blas_thread_init: pthread_create failed for thread 1 of 16: Operation not permitted
```

**Root Cause:** Same seccomp thread limitation, but at runtime. `numpy` (via OpenBLAS) tries to spawn 16 threads on import.

**Fix:** Always run MMORE containers with these flags:
```bash
--pids-limit -1
--security-opt seccomp=unconfined
```

Or for quick one-off commands, also set:
```bash
-e OPENBLAS_NUM_THREADS=1
```

### Milvus Lite DB Locking

**Error:**
```
Open /app/mmore_data/sk_rektor.db failed, the file has been opened by another program
```

**Root Cause:** Milvus Lite uses a single-process SQLite-based file. The running MMORE container holds a lock on it. A second process (e.g., `docker exec`) cannot open the same file.

**Fix:** To inspect the Milvus schema, stop the container first, run a temporary container, then restart:
```bash
docker stop mmore-rag
docker run --rm --entrypoint python \
  --security-opt seccomp=unconfined --pids-limit -1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -v ~/mmore_data:/app/mmore_data \
  mmore-rag -c "from pymilvus import MilvusClient; ..."
docker start mmore-rag
```

---

## 6. RAG API Response Issues

### `context` field returning `null`

**Error:** The API returned:
```json
{ "input": "...", "context": null, "answer": "..." }
```
The answer was correct (meaning retrieval worked), but `context` was always `null`.

**Root Cause:** The LangChain chain produces a dict with keys: `input`, `docs`, `context`, `answer`. The `make_output` function in `pipeline.py` validates through `MMOREOutput` which only keeps `{input, docs, answer}`, dropping the `context` string.

Then `RAGOutput` in `run_rag.py` expects `context`, but since it was dropped, it defaults to `None`.

**Fix** (in `src/mmore/rag/pipeline.py`):
```python
def make_output(x):
    res_dict = MMOREOutput.model_validate(x).model_dump()
    res_dict["answer"] = res_dict["answer"].split("<|im_start|>assistant\n")[-1]
    # Preserve context string from the chain
    if "context" in x:
        res_dict["context"] = x["context"]
    return res_dict
```

---

## 7. Network & Firewall

### Port 8000 not accessible from outside

**Diagnosis:** From a local Windows machine:
```powershell
Test-NetConnection 152.118.31.54 -Port 8000
# TcpTestSucceeded : False
```

**Root Cause:** The server firewall (`ufw` / `iptables`) did not have port 8000 open. The user had no `sudo` access to modify firewall rules.

**Fix:** Requested the server admin to open port 8000/tcp:
```bash
sudo ufw allow 8000/tcp
```

After the admin opened the port:
```powershell
Test-NetConnection 152.118.31.54 -Port 8000
# TcpTestSucceeded : True
```

---

## 8. Final Working Configuration

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app

ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1

COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir -e ".[rag,cpu]" nltk tiktoken uvicorn fastapi "transformers==4.48.0"

ENTRYPOINT ["python", "-m", "mmore"]
```

### Build
```bash
DOCKER_BUILDKIT=0 docker build --no-cache -t mmore-rag .
```
> `DOCKER_BUILDKIT=0` is required on Docker 20.10.8 to avoid BuildKit-related issues.

### Run
```bash
docker run -d --name mmore-rag \
  --pids-limit -1 \
  --security-opt seccomp=unconfined \
  --network host \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore_data:/app/mmore_data \
  -e OPENAI_API_KEY=dummy \
  --restart unless-stopped \
  mmore-rag rag --config-file /app/configs/sk_rektor_rag_docker.yaml
```

### RAG Config (`sk_rektor_rag_docker.yaml`)
```yaml
rag:
  llm:
    llm_name: qwen2.5:latest
    base_url: http://localhost:11434/v1
    max_new_tokens: 1200
    temperature: 0.7
  retriever:
    db:
      uri: /app/mmore_data/sk_rektor.db
      name: sk_rektor_db
    collection_name: sk_rektor_docs
    hybrid_search_weight: 0.5
    k: 5
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base
  system_prompt: |
    Kamu adalah asisten akademik Universitas Indonesia yang menjawab pertanyaan berdasarkan dokumen SK Rektor.

    ATURAN:
    1. Jawab HANYA berdasarkan konteks yang diberikan.
    2. Jika informasi tidak tersedia, katakan: "Maaf, informasi tersebut tidak tersedia dalam dokumen yang saya miliki."
    3. Jika pertanyaan ambigu, minta klarifikasi.
    4. Sebutkan sumber dokumen (nomor SK, pasal, ayat) jika tersedia.
    5. Jawab dalam Bahasa Indonesia yang formal dan ringkas.

    Konteks:
    {context}
mode: api
mode_args:
  endpoint: '/rag'
  port: 8000
  host: '0.0.0.0'
```

---

## 9. API Usage

### Endpoint
| Property | Value |
|----------|-------|
| URL | `http://<SERVER_IP>:8000/rag` |
| Method | `POST` |
| Content-Type | `application/json` |

### cURL Example
```bash
curl -X POST http://152.118.31.54:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": "Siapa rektor UI?",
      "collection_name": "sk_rektor_docs"
    }
  }'
```

### Postman
- Method: POST
- URL: `http://152.118.31.54:8000/rag`
- Body → raw → JSON:
```json
{
  "input": {
    "input": "Siapa rektor UI?",
    "collection_name": "sk_rektor_docs"
  }
}
```

### Example Response
```json
{
  "input": "Siapa rektor UI?",
  "context": "PERATURAN REKTOR UNIVERSITAS INDONESIA NOMOR 16 TAHUN 2025 ...",
  "answer": "Rektor Universitas Indonesia adalah Prof. Dr. Ir. Heri Hermansyah, S.T., M.Eng., IPU."
}
```

### Viewing Container Logs
```bash
# Live logs
docker logs -f mmore-rag

# Last 50 lines
docker logs --tail 50 mmore-rag
```

> **Note:** The API currently does not support streaming. Responses are returned synchronously after retrieval + generation completes.

---

## 10. Milvus Standalone Transition (Database Errors)

### Problem: `milvus-lite` Crashing on Startup
The `index` command silently crashed when initializing Milvus. The error was a silent `SIGKILL` or an unhandled C++ exception inside `milvus-lite`.
**Root Cause:** The embedded `milvus-lite` engine uses an embedded SQLite instance and requires aggressive multithreading (`libgomp1`, `libatomic1`). Even with the dependencies installed, the strict `seccomp` profile of Docker 20.10.8 blocked internal syscalls of the engine, causing a panic/DNS resolution failure (`dns:///`).

### Fix: Migrate to Milvus Standalone
Instead of using the embedded `milvus-lite` engine inside the Python script, we spun up a full **Milvus Standalone** instance using the official `docker-compose.yml`.
1. The compose file spins up 3 containers: `etcd`, `minio`, and `milvus`.
2. The database files are stored locally in the `volumes/` directory on the host.
3. In `configs/sk_rektor_index_docker.yaml` and `configs/sk_rektor_rag_docker.yaml`, the URI was changed from a local `.db` file to:
   ```yaml
   uri: http://localhost:19530
   name: default
   ```

### Follow-up Error: `docker build` Permission Denied (can't stat 'volumes')
Because Milvus runs as `root`, the `volumes/` directory it created was owned by root. Running `docker build .` caused Docker to try to read the entire `~/mmore` directory as the build context, failing on `volumes/`.
**Fix:** Added `volumes/` to `.dockerignore`.

---

## 11. Docker Permissions & File Ownership Hell

### Problem: `rm -rf` Permission Denied
When attempting to clear old extracted data (`~/mmore/data_ingestion/outputs/*`), the user got `Permission denied`.
**Root Cause:** The `mmore-all` Docker container ran as `root`, so all output JSONL files and extracted images produced on the mounted host volume were owned by `root`.

### Failed Fix: Forcing `--user`
Attempting to run the container as `--user "$(id -u):$(id -g)"` broke the entire pipeline:
- `PermissionError: [Errno 13] Permission denied: './profiling_output'`
- `KeyError: 'getpwuid(): uid not found: 1004'` (PyTorch tried to map the UI to `/etc/passwd`)
- `PermissionError: [Errno 13] Permission denied: '/nltk_data'` (NLTK tried writing to root directories)
- HuggingFace cache errors.

**Actual Fix:** Since the MMORE Dockerfile installs dependencies globally as `root` inside the container, trying to run it as a non-root user via `--user` causes compounding errors. The best practice is to **run the container normally (without `--user`)**. To delete the root-owned files later, spin up a tiny alpine container to do the deletion:
```bash
docker run --rm -v ~/mmore/data_ingestion/outputs:/tmp_outputs alpine rm -rf /tmp_outputs/process
```

---

## 12. Data Quality & Processor Enhancements

### A. Removing HTML Boilerplate
**Problem:** `HTMLProcessor` used `markdownify`, which indiscriminately kept navigation menus, sidebars, and footers (e.g., "Home, Tentang Kami, Hubungi"). This polluted the RAG context.
**Fix:** Swapped `markdownify` for `trafilatura`. Trafilatura algorithmically targets the main body/article text of a webpage and completely strips away the boilerplate.

### B. Removing Noisy Image Placeholders
**Problem:** Extracted HTML and PDFs were injecting massive `<attachment>` tags and raw image URLs into the text, taking up valuable token limits.
**Fix:** Set `extract_images: false` in `sk_rektor_process_docker.yaml`. This propagates natively down to Marker-PDF and Trafilatura, forcing them to drop inline images entirely.

### C. Stripping `<think>` Tags from Qwen/DeepSeek
**Problem:** Reasoning models like `qwen2.5:latest` output raw `<think>...</think>` tags containing their Chain-of-Thought process before the actual answer.
**Fix:** Added a regex cleaner in `src/mmore/rag/pipeline.py` inside `make_output(x)` to surgically strip out any `<think>` blocks from the final response, guaranteeing clean API output to the user.

---

## 13. Future Roadmap & To-Do List

- [ ] **Implement Pipeline Profiling & Timing Measurement**
  - Add native timing capabilities (`time.time()` blocks or `langchain.debug = True`) to definitively measure the latency offset of Retrieval (search/reranker latency) vs Generation (LLM token streaming).
- [x] **Implement RAGAS Evaluation**
  - Synthesized a golden Q&A dataset based on Universitas Indonesia facts.
  - Ran the `ragas` evaluation metrics across different Qwen 2.5 models using a local `qwen2.5:7b` judge.
  
  **Initial Evaluation Results (Round 1):**
  | Model | Latency (s) | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
  |---|---|---|---|---|---|
  | qwen2.5:0.5b | 3.67 | 0.0000 | 0.5224 | 0.4000 | 0.4667 |
  | qwen2.5:1.5b | 2.42 | 0.5278 | 0.6716 | 0.4000 | 0.4667 |
  | qwen2.5:3b | 3.65 | 0.5952 | 0.4954 | 0.4000 | 0.4667 |
  | qwen2.5:7b | 6.17 | 0.6595 | 0.3941 | 0.4000 | 0.4667 |

  **Analysis & Next Steps:**
  - **Retriever Bottleneck (Context Recall=0.46):** Context Precision and Recall are exactly identical across all models. This proves that the true bottleneck right now is the **Retrieval** phase (Milvus/Embeddings). The database is only pulling ~46% of the correct facts needed to answer the questions. The LLMs are currently starved for context.
  - **Generator Faithfulness:** Faithfulness correlates perfectly with model size. The 0.5b model completely hallucinates when the context is missing, while the 7b model is the most faithful (0.65). 
  - **Action Item:** The immediate next step is to open `sk_rektor_rag_docker.yaml` and increase the retriever `k` (top-k search results) from 5 to maybe 10 or 15. This will force Milvus to fetch more context chunks and push the Context Recall closer to 1.0!

  **Round 2 Evaluation Results (k=15):**
  | Model | Latency (s) | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
  |---|---|---|---|---|---|
  | qwen3.5:9b | 76.73 | 1.0 | 0.5043 | 1.0000 | 1.0000 |
  | gemma3:4b | 7.94 | 1.0 | 0.7758 | 1.0000 | 1.0000 |
  | gemma3:1B | 8.48 | NaN | 0.5460 | NaN | 1.0000 |
  | ministral-3:3b | 16.07 | NaN | 0.5027 | 1.0000 | 1.0000 |
  | ministral-3:8b | 25.04 | NaN | 0.5718 | NaN | NaN |
  | llama3.2:3b | 8.68 | 0.5 | 0.4420 | 1.0000 | 1.0000 |

  **Analysis (Round 2):**
  - **Success (k=15):** The increase to `k: 15` successfully pushed `Context Recall` and `Context Precision` to a perfect **1.0** for almost all models! This proves our retrieval bottleneck is solved.
  - **NaN Issues:** `NaN` values occur when the Judge LLM fails to output a parsable score (e.g., if it gets confused or outputs reasoning text instead of a numeric score). 
  - **Judge Bias:** Using `gemma3:4b` as a judge for other Gemma models can introduce "self-preference" bias. Reverting to **Qwen 2.5 7B** (at 0.0 temperature) is recommended for the most neutral and stable scoring.
  - **Inference Speed:** `qwen3.5:9b` is significantly slower (76s) than the other models, likely due to internal architecture or quantization overhead in Ollama.

- [ ] **LLM Evaluation: Standalone MMORE vs Built-from-Scratch vLLM**
  - Compare the rigid MMORE LCEL chains against a custom-built, lightweight LangChain+vLLM architecture.
  - Test if a custom architecture yields better performance due to explicit control over overlap metadata and native API Streaming responses.
- [ ] **Stream Response Support**
  - Rewrite `@app.post(endpoint)` in `run_rag.py` to utilize FastAPI's `StreamingResponse` for realtime token streaming on the dashboard, bypassing the static `.invoke()` method.

---

## 14. Architecture Migration Plan: Moving Beyond MMORE

While MMORE provided a solid foundation, its heavy abstraction layers create a "black box" effect that complicates deep architectural debugging, particularly for a research thesis where explainability is paramount. 

To achieve our production goals, we plan to migrate from the out-of-the-box MMORE setup to a **Semi-Custom Python Pipeline** built directly with LangChain and vLLM.

### Key Drivers for Migration:
1. **High Concurrency & Inference Speed (vLLM)**
   * **Current State**: Ollama provides a great developer experience but struggles with handling concurrent user requests efficiently.
   * **Target State**: Switch the deployment engine to `vLLM` to utilize PagedAttention and continuous batching, which allows the AI engine to handle many UI students simultaneously without bottlenecking.
2. **Dynamic Reasoning Control**
   * **Goal**: Provide the ability to toggle "Deep Thinking" modes dynamically. Some simple Q&A tickets don't need reasoning overhead, while complex policy questions do. A custom LangChain pipeline lets us granularly control generation kwargs per route.
3. **Robust Data Extraction & Verification**
   * **Goal**: Move toward a rigorous extraction combo using **Docling** (for layout-aware PDF parsing) and **Trafilatura** (for aggressive HTML cleaning).
   * We need a built-in verification/audit system where extracted text can be manually reviewed or automatically flagged if the extraction confidence is low.
4. **Metadata Filtering (Crucial)**
   * Our RAG system must support precise retrieval filtering (e.g., restricting vector searches strictly to `doc_id` or specific `category` metadata) to avoid hallucinating answers from unrelated academic policies. MMORE's default `retriever.py` wraps this away; a custom Milvus + LangChain retriever gives us direct control over the expression (`expr`) filters.
5. **Native RAGAS Integration**
   * Having direct access to the pipeline steps (Input, Context, Answer) makes logging and scoring these requests with RAGAS much more transparent for the research paper.

