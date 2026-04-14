from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import logging

from pipeline import RAGPipeline
from config import RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# ── Global pipeline instance ──────────────────────────────────────────────────
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown using the modern FastAPI lifespan pattern."""
    global rag_pipeline
    config_path = os.getenv("RAG_CONFIG_PATH", "config_rag.yaml")
    try:
        if os.path.exists(config_path):
            config = RAGConfig.from_yaml(config_path)
            rag_pipeline = RAGPipeline.from_config(config)
            logger.info(f"RAG Pipeline initialized from {config_path}")
        else:
            logger.warning(
                f"Config file {config_path} not found. Pipeline not initialized."
            )
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}", exc_info=True)

    yield  # Application runs here

    # Graceful shutdown (add resource cleanup here if needed in future)
    logger.info("RAG API shutting down.")


app = FastAPI(
    title="MyRAG System API",
    description=(
        "High-performance RAG pipeline using Harrier (dense), "
        "OpenSearch (sparse) with RRF fusion, and Jina-v3 listwise reranker."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    config_override: Optional[Dict[str, Any]] = None
    metadata_filter: Optional[Dict[str, Any]] = None

    model_config = {"json_schema_extra": {"example": {"query": "Apa itu mekanisme penelaahan usulan pembukaan program studi?"}}}


class IngestRequest(BaseModel):
    directory_path: str

    model_config = {"json_schema_extra": {"example": {"directory_path": "/app/data"}}}


class RAGResponse(BaseModel):
    answer: str
    context: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_thought_process(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (Qwen, DeepSeek)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health_check():
    """Returns pipeline and storage connectivity status."""
    return {
        "status": "healthy" if rag_pipeline else "uninitialized",
        "milvus": (
            "connected"
            if rag_pipeline and rag_pipeline.storage
            else "not_connected"
        ),
    }


@app.get("/collections", summary="List indexed collections")
async def list_collections():
    """Returns the collections currently available in the vector store."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    try:
        collections = rag_pipeline.storage.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RAGResponse, summary="Run a RAG query")
async def query_rag(request: QueryRequest):
    """
    Embed the query using dense (Harrier) + sparse (OpenSearch) models,
    fuse results with RRF, rerank with Jina-v3, and generate a grounded answer.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    try:
        result = rag_pipeline.query(
            request.query,
            metadata_filter=request.metadata_filter,
        )
        clean_answer = strip_thought_process(result.answer)

        return {
            "answer": clean_answer,
            "context": result.context,
            "sources": result.sources,
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _background_ingestion(directory: str) -> None:
    """Background task: ingest all documents in *directory* into the vector store."""
    if not rag_pipeline:
        logger.error("Ingestion failed: pipeline not initialized")
        return
    try:
        logger.info(f"Background ingestion started for: {directory}")
        rag_pipeline.ingest(directory=directory)  # keyword arg — matches pipeline signature
        logger.info(f"Background ingestion completed for: {directory}")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}", exc_info=True)


@app.post("/ingest", summary="Ingest a directory of documents")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers document ingestion as a background task.
    Monitor progress with: docker compose logs -f rag-api
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    if not os.path.exists(request.directory_path):
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {request.directory_path}",
        )

    background_tasks.add_task(_background_ingestion, request.directory_path)
    return {
        "status": "ingestion_started",
        "directory": request.directory_path,
        "message": "Check container logs for progress.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
