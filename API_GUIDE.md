# MyRAG API Guide

This guide covers the live API for the MyRAG server.

Base URL:

```text
http://152.118.31.54:8000
```

Swagger UI:

```text
http://152.118.31.54:8000/docs
```

## Quick Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the API and vector store are reachable |
| `GET` | `/collections` | List indexed Milvus collections |
| `GET` | `/ingestion/status` | View ingested files and chunk counts |
| `POST` | `/query` | Run a standard RAG query |
| `POST` | `/query/stream` | Run a streaming RAG query over SSE |
| `POST` | `/ingest` | Start ingestion for a directory or file path |
| `POST` | `/ingestion/upload` | Upload a PDF or HTML file and ingest it |
| `POST` | `/debug/chunks` | Inspect chunks before embedding |
| `POST` | `/debug/retrieve` | Inspect retrieval output before reranking |
| `POST` | `/debug/rerank` | Inspect reranked results before generation |

## Base Notes

- The API listens on port `8000`.
- `query` endpoints accept optional `metadata_filter`.
- The streaming endpoint returns Server-Sent Events.
- Ingestion runs in the background, so the response comes back before processing finishes.
- Normal ingestion writes one job-level chunk snapshot when `save_snapshots` is enabled.
- Reranking is optional in the server deployment. If `retrieval.reranker_model` is `null` in `config_server.yaml`, the API skips reranker use and you do not need to start the reranker container.

## Health And Storage

### `GET /health`
Check whether the API is up and whether the Milvus connection is available.

```bash
curl -X GET http://152.118.31.54:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "milvus": "connected"
}
```

### `GET /collections`
List the collections currently available in Milvus.

```bash
curl -X GET http://152.118.31.54:8000/collections
```

### `GET /ingestion/status`
View the files already ingested by the pipeline.

```bash
curl -X GET http://152.118.31.54:8000/ingestion/status
```

Example response:

```json
{
  "ingested_files": [],
  "count": 0
}
```

## Query

### `POST /query`
Run a standard RAG query. If reranking is enabled in the active config, the server retrieves candidates, reranks them, and returns a grounded answer.

Request body:

```json
{
  "query": "Apa itu mekanisme penelaahan usulan pembukaan program studi?",
  "metadata_filter": null,
  "config_override": null
}
```

Example:

```bash
curl -X POST http://152.118.31.54:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Apa itu mekanisme penelaahan usulan pembukaan program studi?"}'
```

Response shape:

```json
{
  "answer": "...",
  "context": "...",
  "sources": [],
  "metadata": {
    "confidence_score": 0.0
  }
}
```

### `POST /query/stream`
Run the same query flow, but stream output as SSE events.

```bash
curl -N -X POST http://152.118.31.54:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Jelaskan langkah-langkah akreditasi."}'
```

Use `-N` so `curl` does not buffer the stream.

## Ingestion

### `POST /ingest`
Trigger ingestion for a directory path or a single file path. The job runs in the background.

```bash
curl -X POST http://152.118.31.54:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path":"/app/data"}'
```

Example response:

```json
{
  "status": "ingestion_started",
  "directory": "/app/data",
  "message": "Check container logs for progress."
}
```

Ingestion is incremental and content-aware:

- New or modified files are parsed, chunked, embedded, and inserted into Milvus.
- Unchanged files are skipped without parsing, chunking, or embedding.
- Files with duplicate byte-for-byte content are skipped even when the filename or path is different.
- Duplicate files are recorded as aliases of the canonical document in `storage/ingestion_state.json`.

Use `/ingest` for normal incremental updates. If parsing or chunking behavior changes and the same source files need to be rebuilt, do not wipe the live Milvus collection first. Use the CLI-only `rebuild-index` workflow to build a shadow collection with a fresh state file, validate it, then promote it intentionally.

When `save_snapshots: true`, each `/ingest` call writes one snapshot file:

```text
storage/snapshots/ingest_job_<timestamp>.json
```

The snapshot is a debug artifact, not the retrieval source of truth. Milvus remains the source used by `/query` and debug retrieval endpoints.

Snapshot entries for processed files include the actual chunk text:

```json
{
  "status": "new",
  "path": "/app/data/example.pdf",
  "doc_id": "doc_ab12cd34ef56",
  "chunk_count": 3,
  "chunks": [
    {
      "chunk_index": 0,
      "text": "Actual chunk text...",
      "page_number": 1,
      "metadata": {}
    }
  ]
}
```

Snapshot entries for unchanged or duplicate files are manifest-only for efficiency. They show the reason and, for duplicates, the canonical document:

```json
{
  "status": "duplicate",
  "path": "/app/data/renamed-example.pdf",
  "canonical_path": "/app/data/example.pdf",
  "canonical_doc_id": "doc_ab12cd34ef56",
  "reason": "same content as canonical file"
}
```

### `POST /ingestion/upload`
Upload one document and immediately queue it for ingestion.

```bash
curl -X POST http://152.118.31.54:8000/ingestion/upload \
  -F "file=@/path/to/document.pdf"
```

Supported uploads are PDF and HTML documents.

## Debugging

### `POST /debug/chunks`
Inspect chunks after parsing and chunking, before embedding.

```bash
curl -X POST http://152.118.31.54:8000/debug/chunks \
  -H "Content-Type: application/json" \
  -d '{"directory_path":"/app/data","save_to_file":false,"output_format":"json"}'
```

Useful when you want to verify chunk boundaries before embedding. PDFs use hierarchical chunking; HTML uses standard overlapping text chunks.

### `POST /debug/retrieve`
Inspect the raw retrieval output after hybrid search and RRF fusion, before reranking.

```bash
curl -X POST http://152.118.31.54:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"testing query","k":20,"metadata_filter":null}'
```

### `POST /debug/rerank`
Inspect the reranked candidates before the LLM answer is generated.
This endpoint only works when the reranker service is deployed and the active config points to it.

```bash
curl -X POST http://152.118.31.54:8000/debug/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"testing query","k":20,"rerank_top_k":5}'
```

## Practical Tips

- Use `/debug/chunks` when you want to confirm how the current chunker behaves on PDFs and HTML.
- Use `/debug/retrieve` when retrieval looks weak but chunking looks fine.
- Use `/debug/rerank` when good chunks are being found but the final ordering looks off.
- For browser testing, open `http://152.118.31.54:8000/docs`.
