# Current System State Handover: Incremental Ingestion, Streaming, and File Management

## 1. Latest Features (Implemented April 2026)

The RAG system has been upgraded with production-grade data management and user experience features:

- **Incremental Ingestion**:
    - Uses **MD5 hashing** to detect file changes.
    - State is tracked in `storage/ingestion_state.json`.
    - Automatically deletes stale chunks from Milvus before updating modified files.
- **Streaming Generation**:
    - New `/query/stream` endpoint using **SSE (Server-Sent Events)**.
    - Returns sources first, then a stream of tokens.
- **File Upload API**:
    - New `/ingestion/upload` endpoint for direct document uploads.
    - Files are saved to the mounted `./uploads` directory and ingested in the background.
- **Status Dashboard**:
    - New `/ingestion/status` endpoint to see all indexed files and hashes.

## 2. Technical Decisions & Best Practices

- **FileSystem + JSON State**: We chose a minimalist state management approach (local JSON) instead of a full SQL database to keep the Docker infrastructure lean and portable for research server use.
- **Endpoint Protection**: New features were added as separate endpoints (`/query/stream`, `/ingestion/upload`) to ensure existing synchronous workflows were not disrupted.
- **Dockerized Uploads**: Added a new volume mount for `./uploads` with Read-Write access.
- **Robustness**: Proactively fixed `NameError` issues by ensuring `os` and `pathlib.Path` imports are present across core modules.

## 3. GPU Memory Tracking (Persistent Context)

The system continues to use **Lazy Loading** and **8-bit Quantization** to stay within the 8GB VRAM limit:
- **Embedding Models**: Harrier (Dense) and OpenSearch (Sparse) are loaded on-demand.
- **Reranker**: Jina-v3 is loaded/unloaded per query.
- **LLM**: Typically runs in 4-bit/8-bit to share VRAM with the retrieval models.

## 4. Verification & Testing

- **Test Script**: `scripts/test_upgrades.py` provides a unified way to test status, uploads, and streaming.
- **Logs**: Ingestion progress can be monitored via `docker compose logs -f rag-api`. Unchanged files will be skipped with an explicit log message.

## 5. Next Steps for Future Agents

- **Frontend Integration**: Build a UI that consumes the new SSE stream and file upload endpoints.
- **Deletions**: Currently, the system handles additions and modifications. Implementing a "clean-up" task for files deleted from the filesystem (but still in Milvus) is a potential future task.
- **Semantic Routing**: Future plans include a semantic FAQ router to short-circuit the LLM for high-confidence queries.