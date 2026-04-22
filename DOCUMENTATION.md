# MyRAG Implementation Documentation

## 1. Overview

MyRAG is a modular, research-oriented Retrieval-Augmented Generation (RAG) pipeline. Its primary goal is to provide a transparent, debuggable, and extensible system for transforming unstructured documents (PDFs, HTML) into a high-precision QA system.

Unlike "black-box" RAG frameworks, MyRAG prioritizes **explainability**, allowing researchers to trace exactly which chunks were retrieved and why, and evaluate the quality of each stage using industry-standard metrics.

---

## 2. Technical Stack

### A. Document Parsing & Chunking

To ensure high-quality text extraction and structural preservation, we use a sophisticated ingestion strategy:

- **Parsing**:
  - **All Formats (PDF, HTML, etc.)**: `Docling` (by IBM). Docling is used exclusively across all ingestion formats because it natively identifies layout structures (tables, headers, formulas) and outputs a strictly typed `DoclingDocument`. 
  - *Note on Trafilatura*: We previously used `Trafilatura` for HTML boilerplate removal. However, because our advanced `HybridChunker` specifically requires the hierarchical node metadata inside a `DoclingDocument` to function correctly, we deprecated custom flat-text parsers. Docling handles HTML natively while preserving the required structural geometry.
- **Chunking**:
  - We use `docling.chunking.HierarchicalChunker`. 
  - This focuses strictly on the document's organizational hierarchy (headers, sections, lists) to produce segments that map human-logical boundaries.
  - Each chunk is **contextualized** (prepended with its heading hierarchy), providing critical context for retrieval.
  - `ChunkRecord`s include a **breadcrumb** (e.g., `"Introduction > Methods > Data Collection"`) and exact **page numbers**.
  - **Intermediate Snapshots**: For debugging, the system can save individual chunks as JSON files to `storage/snapshots` before they are embedded. This is controlled by `ingestion.save_snapshots` in the config.
  - **Network Configuration**: To bypass host firewall (UFW) blockers that often drop packets from the Docker bridge to the host IP, we use `network_mode: host`. This removes Docker-network isolation for this container and allows it to talk to `localhost:11434` (Ollama) and `localhost:19530` (Milvus) with zero firewall interference.

### B. Embedding & Vector Storage

We implement a **Hybrid Retrieval** strategy using a **Dual-Routing** architecture to maximize both accuracy and speed.

#### 1. Dense Strategy: `microsoft/harrier-oss-v1-0.6b`

A decoder-only multilingual embedding model (Qwen-based) using last-token pooling.

- **Ingestion Path**: Documents are embedded plainly using `embed_documents()` - no instruction prompt.
- **Query Path**: Queries use `embed_query()` which automatically applies the `web_search_query` instruction prompt. This alignment is critical: the model was trained to identify relevant passages specifically when prompted with a task instruction.

#### 2. Sparse Strategy: `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte`

An **Asymmetric, Inference-Free** learned sparse retriever.

- **Ingestion Path**: Uses the **Full Neural Model** (`model.encode_documents`) to expand documents with latent terms and importance weights.
- **Query Path**: Uses the **Inference-Free Path** (`model.encode_queries`). It utilizes a pre-computed Tokenizer + IDF weight lookup table. This removes the need for a GPU forward pass during search, resulting in near-zero query latency for sparse retrieval.

### C. Retrieval Fusion: Reciprocal Rank Fusion (RRF)

After both searches return their top-k candidates, we fuse them using **Custom RRF**:

```
score(d) = sum(1 / (k + rank_i))
```

- `k = 60` (the standard constant from the original RRF paper).
- Dense and sparse rank lists are merged - documents appearing in both lists get a higher combined score.
- The result is a single sorted candidate pool (typically top-50) passed to the reranker.

### D. Reranking

For late-stage precision, we use a **Listwise Cross-Encoder Reranker**:

- **Model**: `jinaai/jina-reranker-v3-GGUF:Q5_K_M` served through llama.cpp. This keeps the reranker on-GPU while cutting VRAM compared with the full FP32 transformer path.
- **Process**: The retriever fetches a wide candidate pool (Top-50). The reranker performs a deep pairwise comparison between the query and each candidate, re-sorting them by semantic relevance.
- **Top-K Slicing**: After reranking, only the top `rerank_top_k` documents (default: 5) are passed to the LLM. This prevents token-window overflow while ensuring the LLM receives the highest-quality context.
- **Why this is the memory hotspot**: the reranker is listwise and long-context. VRAM usage is dominated by the total query + chunk tokens, attention buffers, and activations, not just the 0.6B parameter count. On GTX 1080-class GPUs, the current FP32 transformer path can easily reach multi-GB usage per query.
- **What changed**: dense/sparse embeddings are loaded once and kept resident. The reranker is no longer loaded in-process; it is a dedicated llama.cpp service over HTTP.

### E. Generation

- **LLM**: Local models served via **vLLM** or **Ollama** (accessed via OpenAI-compatible API).
- **Grounding & Attribution**:
  - The system uses a strict system prompt that forces the LLM to rely ONLY on the provided context.
  - Each chunk is prepended with its breadcrumb: `Source [Breadcrumb]: Text`.
  - The system returns an explicit list of sources (breadcrumb, filename, page) used in the answer.
  - `<think>...</think>` tags from reasoning models (e.g., Qwen, DeepSeek) are automatically stripped.

### F. Incremental Ingestion & State Management

To avoid redundant processing and ensure index integrity, the system implements **Incremental Ingestion**:

- **Content Change Detection**: Every file is hashed using **MD5**. The system compares the current file hash against a local registry before processing.
- **State Registry**: A JSON file (`storage/ingestion_state.json`) tracks every ingested file, its hash, and the number of chunks generated.
- **Deduplication**: 
    - **UNCHANGED**: If hashes match, the file is skipped entirely.
    - **MODIFIED**: If a hash changes, the system performs a `delete_by_source` operation in Milvus to remove stale chunks before re-indexing the new content.
- **File Uploads**: A dedicated directory (`/app/uploads`) is mounted to the container to handle documents uploaded via the API, which are then queued for immediate ingestion.

### G. Streaming Generation

For a responsive user experience, the system supports real-time token streaming:

- **Protocol**: Server-Sent Events (SSE).
- **Format**:
    - **Type: `sources`**: The first message sent contains the full list of retrieved document metadata.
    - **Type: `token`**: Subsequent messages contain individual text tokens as they are emitted by the LLM.
- **Implementation**: The pipeline uses the `stream=True` parameter in the OpenAI-compatible client, yielding chunks directly to the FastAPI `StreamingResponse`.

### H. Evaluation & Observability

- **Framework**: `RAGAS`.
- **Metrics**: Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- **Failure Categorization**: The evaluator automatically categorizes failures into:
  - **Retrieval Failure**: Relevant info was not in the top-50.
  - **Reranking Failure**: Relevant info was in top-50 but ranked too low.
  - **Generation Failure**: Correct context was present, but the LLM failed to use it.
- **Synthetic QA**: A module that uses the LLM to generate "Ground Truth" Q&A pairs from your documents.

---

## 3. The Dual-Routing Map

![alt text](<assets/routing-map.png>)

## 4. Pipeline Data Flow

### Ingestion Flow
`Raw Files` -> `Docling Parsing` -> `Hierarchical Chunking` -> `Per-file Dense & Sparse Embedding` -> `Milvus Storage (with Breadcrumbs & Page Nos)`

### Query Flow
`User Query` -> `Dual Embedding` -> `Milvus Hybrid Search` -> `Metadata Filtering (Optional)` -> `RRF (k=60)` -> `Top-50 Candidates` -> `Jina Reranking` -> `Top-5 Context` -> `Grounded Generation` -> `Answer + Sources`

---

## 5. Configuration Reference

All configuration is driven by `config_rag.yaml` (local) or `config_server.yaml` (Docker server).

### Key Parameters

| Section | Parameter | Default | Description |
|---|---|---|---|
| `ingestion` | `chunk_size` | `512` | Max tokens per chunk (Harrier tokenizer) |
| `ingestion` | `pdf_parser` | `docling` | Parser for PDF files |
| `embedding` | `device` | `cuda:0` | Default GPU for embedding models |
| `embedding` | `dense_device` | `cuda:0` | Explicit GPU for Dense model |
| `embedding` | `sparse_device` | `cpu` | Device for Sparse query encoding |
| `retrieval` | `reranker_endpoint` | `http://127.0.0.1:8012/v1/rerank` | llama.cpp reranking endpoint |
| `retrieval` | `k` | `50` | Candidate pool size fetched from Milvus |
| `retrieval` | `rerank_top_k` | `5` | Docs passed to the LLM after reranking |
| `retrieval` | `reranker_model` | `jinaai/jina-reranker-v3-GGUF:Q5_K_M` | Reranker model name/path used by llama.cpp |
| `generation` | `llm_endpoint` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint |
| `generation` | `model_name` | `llama-3-8b` | Model served by vLLM/Ollama |

---

## 6. API Reference

The service exposes a FastAPI application on port `8000`.

### `GET /health`
Verifies pipeline and vector database connectivity.
```json
{"status": "healthy", "milvus": "connected"}
```

### `GET /collections`
Lists all indexed collections in the vector store. Useful for debugging.
```json
{"collections": ["documents"]}
```

### `POST /query`
The primary endpoint for retrieving context and generating answers.
- **Request Body**:
  ```json
  {
    "query": "Apa itu mekanisme penelaahan usulan pembukaan program studi?",
    "metadata_filter": {"doc_id": "sk_rektor_001"},
    "config_override": {}
  }
  ```
- **Response**: Returns the answer, combined context, and an array of source documents with breadcrumbs, filenames, and page numbers.

### `POST /query/stream`
Streaming version of the RAG query. Returns tokens as they are generated.
- **Request Body**: Same as `/query`.
- **Response**: SSE stream of JSON objects:
  ```json
  {"type": "sources", "content": [...]}
  {"type": "token", "content": "Hello"}
  ```

### `POST /ingest`
Triggers a background ingestion process for a directory.
- **Request Body**:
  ```json
  {"directory_path": "/app/data"}
  ```
- **Response**: `{"status": "ingestion_started", "directory": "/app/data"}`
- **Incremental Logic**: If `ingestion.incremental` is true, it only processes new/modified files.

### `POST /ingestion/upload`
Uploads a single file (PDF/HTML) and triggers ingestion.
- **Multipart Form**: `file` (the document to upload).
- **Processing**: Saves to `/app/uploads` and runs the incremental ingestion loop.

### `GET /ingestion/status`
Returns a dashboard of all currently ingested files.
- **Response**: List of file paths, MD5 hashes, and chunk counts.


### Interactive Docs
Visit `http://<SERVER_IP>:8000/docs` for the full Swagger UI.

### Debug Endpoints
The system includes several debugging endpoints to inspect the pipeline at different stages:

**`POST /debug/chunks`** - View chunks after chunking but before embedding
- Request body: `{"directory_path": "/app/data", "save_to_file": true, "output_format": "json"}`
- Response: Array of chunk objects with detailed metadata (text, doc_id, breadcrumb, page_number, etc.)

**`POST /debug/retrieve`** - View retrieval results after RRF fusion but before reranking
- Request body: `{"query": "your query", "k": 20, "metadata_filter": {}}`
- Response: Retrieved documents with RRF scores

**`POST /debug/rerank`** - View reranking results after Jina reranking but before LLM generation
- Request body: `{"query": "your query", "k": 20, "rerank_top_k": 5}`
- Response: Reranked documents with updated scores

## 7. CLI Usage (Extended)

### 7.7 Debugging Commands

**`python cli.py inspect-chunks`** - Inspect chunks before embedding
```bash
python cli.py inspect-chunks --config config_rag.yaml --directory ./data --show-stats
```
Options:
- `--output-file`: Save chunks to a file
- `--show-stats`: Show chunking statistics
- `--filter-keyword`: Filter chunks containing specific keyword

**`python cli.py debug-query`** - Debug query with full pipeline inspection
```bash
python cli.py debug-query --config config_rag.yaml --query "Your query" --show-stages --output-format detailed
```
Options:
- `--show-stages`: Show results at each pipeline stage
- `--output-format`: Output format (json, text, detailed)

These commands allow you to inspect the intermediate results of the RAG pipeline without triggering the LLM generation.
---

## 7. CLI Usage

### 7.1 Ingesting Data
```bash
python cli.py ingest --config config_rag.yaml --directory ./my_docs
```

### 7.2 Standard QA
```bash
python cli.py query --config config_rag.yaml --query "What is the result of the study?"
```

### 7.3 Document-Specific Search
```bash
python cli.py query --config config_rag.yaml --query "..." --doc-ids doc_001 doc_005
```

### 7.4 Keyword Debugging
```bash
# Find all chunks containing a word
python cli.py find-keyword --config config_rag.yaml --keyword "neural network"

# Trace if a specific query's results contained a required keyword
python cli.py trace --config config_rag.yaml --query "..." --check-keyword "activation"
```

### 7.5 Evaluation
```bash
python cli.py eval --config config_rag.yaml --synthetic --paths ./data/doc.pdf
```

---

## 8. Extensibility Guide

### Adding a New Parser
1. Create a class inheriting from `BaseParser` in `ingestion/`.
2. Implement `extract()` and `accepts_extension()`.
3. Register it in `IngestionPipeline.process_file()`.

### Changing the Embedding Model
1. Update `embedding.dense_model` or `embedding.sparse_model` in `config_rag.yaml`.
2. Re-run the `ingest` command to rebuild the Milvus index.

### Adding New Evaluation Metrics
1. Add the metric class from `ragas.metrics` to the `metric_map` in `evaluation/evaluator.py`.
2. Add the metric name to the `metrics` list in `config_rag.yaml`.

---

## 9. Stability & Performance Tuning (Lessons Learned)

The following optimizations were implemented to stabilize the pipeline for production use on the research server.

### A. The "Silent" OCR Failure (`libGL.so.1`)
- **Problem**: In `python:slim` images, OCR engines like `easyocr` or `rapidocr` fail to import because `opencv-python` looks for graphical libraries (`libGL.so.1`) that don't exist in lean containers.
- **Fix**: Used `opencv-python-headless`. This version is optimized for server environments and removes all GUI/OpenGL dependencies.
- **Engine Preference**: `rapidocr-onnxruntime` is used as a lightweight CPU/GPU fallback, while `easyocr` is preferred for native PyTorch/CUDA acceleration on the GTX 1080.

### B. PyMilvus 2.5.0 API Migration
- **Problem**: Upgrading to Milvus 2.5.0 (to fix legacy `pkg_resources` bugs) introduced breaking keyword changes in the Python SDK.
- **Fixes**:
  - Renamed `param` to `search_params` in `MilvusClient.search()`.
  - Renamed `expr` to `filter` for metadata filtering.
  - Removed `setuptools` version pinning as PyMilvus 2.5+ no longer requires the deprecated `pkg_resources` module.

### C. CUDA Out-of-Memory (OOM) Management
- **Problem**: The SPLADE sparse embedding model (`opensearch-v3-gte`) has a 30,522-dimension output. Batch sizes of 32 exceed the 8GB VRAM limit of the GTX 1080 when running dual-models (Harrier + SPLADE).
- **Fix**: Reduced `embedding.batch_size` to `16` (or `4` for maximum safety). This reduces the activation matrix memory pressure during the SpladePooling stage.

| **Issue** | **Root Cause** | **Resolution** |
|---|---|---|
| `CUDA out of memory` during ingestion | SPLADE model vocabulary expansion (30k dims) is too large for batch size 32 | Lower `batch_size` to `16` or `4` in `config_server.yaml` |
| **Increased Query Latency** | Model reload churn to save VRAM | This is no longer used. Dense/sparse embeddings stay loaded, and the reranker is a separate service. |

### D. GPU Memory Competition Issue
- **Problem**: When running the RAG system in Docker with GPU access, the reranker alone can consume most of GPU 1. In older deployments, Ollama could also compete for VRAM, but the reranker is the stage that explains the 7 GB spike you observed.
- **Solution**:
  1. **GPU Isolation**: Assign separate GPUs for embedding/reranking vs LLM inference in docker-compose.yml:
     - GPU 0: Embedding models (Harrier, OpenSearch sparse)
     - GPU 1: Dedicated GGUF reranker service
  2. **Memory Management**: Keep embeddings resident; remove unload/reload churn
  3. **Environment Variable**: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True only if you still see fragmentation in other GPU workloads

### F. Persistent Models
- **Mechanism**: Dense and sparse embedding models are loaded at startup and stay resident for the lifetime of the process.
- **Latency Impact**: Query latency is lower because there is no unload/reload churn between requests.
- **Optimization**: The reranker is the only remaining memory-sensitive stage; run it as a dedicated GGUF service.

- **Fix**: Centralized all parsing (PDF, HTML, DOCX) into the `Docling` `DocumentConverter`. This ensures every chunk retains its structural metadata (breadcrumbs) which is vital for the RAG reranking stage.

### G. Multi-GPU Distribution Strategy
To support 8GB VRAM cards alongside a local LLM, the system implements **Spatial Deconfliction**:
- **GPU 0**: Dedicated to Dense Embedding (Harrier).
- **GPU 1**: Dedicated to the GGUF reranker service.
- **CPU Offloading**: Sparse query encoding is forced to CPU to save VRAM for the dense reranking pass.
- **Explicit Cleanup**: There is no embedding unload path anymore.

---

## 10. Future Roadmap & Planned Features

### Semantic FAQ Router (Intent Routing / Guardrails)
To improve performance, reduce LLM costs, and ensure 100% accuracy for strictly administrative queries (e.g., "lost ID card", "leave of absence steps"), an **Intent Router** is planned to bypass the LLM entirely for high-confidence queries.

#### Architecture Proposal:
1. **FAQ Vector Collection**: A new `faq_collection` in Milvus storing anticipated questions (embedded using `Harrier`) alongside their verified, hardcoded markdown answers in metadata.
2. **Pre-Retrieval Interception**:
   - The user's query is embedded.
   - We perform a cosine similarity search against `faq_collection`.
   - **Threshold Match (> 0.90)**: The semantic router SHORT-CIRCUITS the pipeline. The verified answer is returned instantly (~0.1s latency). No chunks are fetched, Jina is not invoked, and the generation LLM is completely bypassed.
   - **Fallback (< 0.90)**: The query proceeds through the normal Dual-Routing architecture (Dense + Sparse -> RRF -> Reranker -> LLM).
3. **Automated Bootstrapping**: Utilizing the existing `SyntheticQAGenerator` module to crawl regulatory PDFs, generate anticipated Q&A pairs, and stage them for human review before insertion into the `faq_collection`.

---

## 11. Known Issues & Configuration Discrepancies

As of April 2026, there are several areas where the configuration files (`config_rag.yaml`, `config_server.yaml`) do not yet fully connect to the underlying implementation. These are tracked as critical technical debt:

### A. Chunker Configuration (`chunk_size` & `chunk_overlap`)
- **Issue**: The `HierarchicalChunker` (based on Docling) in `ingestion/chunker.py` currently ignores the `chunk_size` and `chunk_overlap` parameters passed to its constructor.
- **Impact**: Changing `chunk_size` in your YAML files will not actually change the size of the chunks generated during ingestion. The chunker currently relies strictly on document structural boundaries (headings, paragraphs).
- **Planned Fix**: Initialize the `HierarchicalChunker` with a `HuggingFaceTokenizer` and a `max_tokens` constraint to bridge this gap.

### B. Embedding Model Context Limits
- **Issue**: Most embedding models (Dense and Sparse) have internal token limits (typically 512 or 8192). 
- **Impact**: If you manually set a very large `chunk_size` (e.g., 8,000) that the embedding model cannot handle, the model will truncate the text before generating the vector. This means the "middle" and "end" of your 8,000-token chunk will not be searchable.
- **Verification**: Always cross-reference your `chunk_size` with the `max_seq_length` of your dense/sparse models.

### C. Reranker Context vs. Server Context
- **Issue**: The reranker model supports 8,192 tokens, but the `llama.cpp` server defaults to a smaller context window if not explicitly overridden. 
- **Impact**: Even if your chunks are 8k tokens, the reranker might only "see" the first 512 of them.
- **Fix**: The `-c 8192` flag in `docker-compose.yml` addresses this.

### D. Ingestion Failure Isolation
- **Current Behavior**: Ingestion is file-scoped. Each file is parsed, embedded, inserted, and marked successful independently.
- **Retry Behavior**: Dense and sparse embedding batches retry with smaller batch sizes after failures such as CUDA OOM. If one file still fails, the run logs that file as failed and continues with the remaining files.
- **Summary Logs**: Each run logs indexed chunks, successful files, failed files, skipped unchanged files, and per-file failure details.

