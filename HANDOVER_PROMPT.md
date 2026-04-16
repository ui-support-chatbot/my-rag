# Handover Prompt: RAG System Debugging Enhancements

## Summary of Work Completed

I have successfully implemented comprehensive debugging enhancements for the RAG system to address the GPU memory issues and provide better visibility into the pipeline stages.

## Problems Identified and Solved

### 1. GPU Memory Issues
**Problem**: The RAG system was experiencing CUDA out of memory errors due to:
- Competition between RAG embedding/reranking models and Ollama LLM for GPU memory
- Memory fragmentation in PyTorch
- Lack of proper GPU memory cleanup between operations

**Solutions Implemented**:
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable to docker-compose.yml to reduce memory fragmentation
- Added explicit `torch.cuda.empty_cache()` calls after intensive operations in:
  - `embedding/dense.py` (after document and query embedding)
  - `embedding/sparse.py` (after document and query embedding)
  - `retrieval/retriever.py` (after reranking)
  - `pipeline.py` (after query processing)
  - `api.py` (after each API query request)

### 2. Missing Debugging Capabilities
**Problem**: No way to inspect intermediate pipeline stages (chunks, retrieval results, reranked results) without triggering LLM generation.

**Solutions Implemented**:
- Added three new API endpoints:
  - `POST /debug/chunks` - View chunks after chunking but before embedding
  - `POST /debug/retrieve` - View retrieval results after RRF fusion but before reranking
  - `POST /debug/rerank` - View reranked results after Jina reranking but before LLM generation
- Added two new CLI commands:
  - `python cli.py inspect-chunks` - Inspect chunks with options for stats and filtering
  - `python cli.py debug-query` - Debug queries with full pipeline inspection
- Enhanced pipeline with `save_chunks_before_embedding()` method to save chunks to file before embedding
- Added detailed data structures with comprehensive metadata (doc_id, breadcrumb, page_number, etc.)

### 3. Technical Issues Encountered and Fixed

**Issue 1: Pydantic Validation Error**
- **Problem**: `ValidationError` for `RetrievedDocInfo` - `page_number` field expected integer but received `None`
- **Root Cause**: Some retrieved documents had `None` values for page_number in metadata
- **Fix**: Updated API endpoints to handle None values with `doc.metadata.get("page_number") or 0`

**Issue 2: Missing OCR Dependencies**
- **Problem**: Docling OCR engines not available - "No OCR engine found" warnings
- **Root Cause**: Missing tesseract-ocr and related dependencies in Docker image
- **Fix**: Added OCR dependencies to Dockerfile:
  ```
  RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libtesseract-dev \
      libleptonica-dev \
      && rm -rf /var/lib/apt/lists/*
  ```

## Files Modified

1. **API Endpoints and Debugging Features**:
   - `api.py` - Added debug endpoints and fixed Pydantic validation
   - `pipeline.py` - Added `save_chunks_before_embedding()` method
   - `cli.py` - Added `inspect-chunks` and `debug-query` CLI commands

2. **Documentation**:
   - `DOCUMENTATION.md` - Added sections for new debug endpoints and CLI usage
   - `HANDOVER_PROMPT.md` - This file (summary of work done)

3. **Infrastructure**:
   - `docker-compose.yml` - Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable
   - `Dockerfile` - Added OCR dependencies for document processing

## How to Use the New Debugging Features

### API Endpoints (after restarting Docker):
1. View chunks before embedding:
   ```bash
   curl -X POST http://localhost:8000/debug/chunks \
     -H "Content-Type: application/json" \
     -d '{
       "directory_path": "./data",
       "save_to_file": true,
       "output_format": "json"
     }'
   ```

2. View retrieval results (before reranking):
   ```bash
   curl -X POST http://localhost:8000/debug/retrieve \
     -H "Content-Type: application/json" \
     -d '{
       "query": "your query here",
       "k": 10
     }'
   ```

3. View reranked results (before LLM):
   ```bash
   curl -X POST http://localhost:8000/debug/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "your query here",
       "k": 10,
       "rerank_top_k": 5
     }'
   ```

### CLI Commands:
1. Inspect chunks:
   ```bash
   python cli.py inspect-chunks --config config_rag.yaml --directory ./data --show-stats
   ```

2. Debug a query with full pipeline inspection:
   ```bash
   python cli.py debug-query --config config_rag.yaml --query "your query" --show-stages --output-format detailed
   ```

### Web Interface:
Visit `http://localhost:8000/docs` to see all endpoints including the new debug ones with interactive testing.

## Deployment Instructions

To apply these changes:

1. **Stop existing containers**:
   ```bash
   docker-compose down
   ```

2. **Rebuild the API service** (to include code changes):
   ```bash
   docker-compose build rag-api
   ```

3. **Start services with new configuration**:
   ```bash
   docker-compose up -d
   ```

4. **Wait for services to start** (check with `docker-compose ps`)

5. **Test the new endpoints** using the examples above

## Verification

After deployment, you should be able to:
- Access the new debug endpoints without 404 errors
- Use the new CLI commands without errors
- See improved GPU memory usage (less fragmentation)
- Have OCR functionality working for PDF processing

The existing indexed data in Milvus remains intact and compatible with these changes - no re-ingestion is required.