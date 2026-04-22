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
Run a standard RAG query. The server retrieves candidates, reranks them, and returns a grounded answer.

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

Useful when you want to verify whether hierarchical chunking is producing the size and boundaries you expect.

### `POST /debug/retrieve`
Inspect the raw retrieval output after hybrid search and RRF fusion, before reranking.

```bash
curl -X POST http://152.118.31.54:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"testing query","k":20,"metadata_filter":null}'
```

### `POST /debug/rerank`
Inspect the reranked candidates before the LLM answer is generated.

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
