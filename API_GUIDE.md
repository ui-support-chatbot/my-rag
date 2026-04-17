# MyRAG API Guide

This guide provides a comprehensive overview of all available API endpoints for the MyRAG system, including descriptions and `curl` usage examples.

## 1. Core Endpoints

### `GET /health`
**Description**: Verifies that the API is running and connected to the Milvus vector database.
```bash
curl -X GET http://localhost:8000/health
```

### `GET /collections`
**Description**: Lists all collections currently indexed in the Milvus vector database.
```bash
curl -X GET http://localhost:8000/collections
```

---

## 2. Query & Generation

### `POST /query`
**Description**: Performs a standard, synchronous RAG query. It retrieves context, reranks, and generates a grounded answer.
**Request Body**:
```json
{
  "query": "Apa itu syarat pembukaan program studi?",
  "metadata_filter": null
}
```
**Example**:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Apa itu syarat pembukaan program studi?"}'
```

### `POST /query/stream`
**Description**: Performs a RAG query with real-time token streaming using Server-Sent Events (SSE).
**Example**:
```bash
curl -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "Jelaskan langkah-langkah akreditasi."}' \
     --no-buffer
```
*Note: Use `--no-buffer` to see tokens as they arrive.*

---

## 3. Ingestion & File Management

### `POST /ingest`
**Description**: Triggers ingestion for documents in a specified directory path. Supports **Incremental Ingestion** (skips unchanged files).
**Example**:
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"directory_path": "/app/data"}'
```

### `POST /ingestion/upload`
**Description**: Uploads a single document (PDF or HTML) via multipart form and triggers an immediate incremental ingestion for that file.
**Example**:
```bash
curl -X POST http://localhost:8000/ingestion/upload \
     -F "file=@/path/to/your/document.pdf"
```

### `GET /ingestion/status`
**Description**: Returns a dashboard of all currently ingested files, including their MD5 hashes and chunk counts.
**Example**:
```bash
curl -X GET http://localhost:8000/ingestion/status
```

---

## 4. Debugging & Inspection

### `POST /debug/chunks`
**Description**: Processes a directory and returns the generated chunks *before* they are embedded. Useful for testing chunking logic.
**Example**:
```bash
curl -X POST http://localhost:8000/debug/chunks \
     -H "Content-Type: application/json" \
     -d '{"directory_path": "/app/data", "save_to_file": false}'
```

### `POST /debug/retrieve`
**Description**: Returns the raw documents retrieved from Milvus after RRF fusion but **before** reranking.
**Example**:
```bash
curl -X POST http://localhost:8000/debug/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query": "testing query", "k": 20}'
```

### `POST /debug/rerank`
**Description**: Returns the top documents **after** the Jina-v3 reranker has processed them.
**Example**:
```bash
curl -X POST http://localhost:8000/debug/rerank \
     -H "Content-Type: application/json" \
     -d '{"query": "testing query", "k": 20, "rerank_top_k": 5}'
```

---

## 5. Interaction via Swagger UI
For an interactive experience and a full OpenAPI specification, visit the Swagger UI in your browser:
`http://localhost:8000/docs`
