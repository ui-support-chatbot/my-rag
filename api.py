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

    # Retry loop — Milvus may still be warming up even after its healthcheck
    # passes. We retry up to 5 times (50 seconds total) before giving up.
    max_attempts = 5
    retry_delay = 10  # seconds

    for attempt in range(1, max_attempts + 1):
        try:
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                break

            config = RAGConfig.from_yaml(config_path)
            rag_pipeline = RAGPipeline.from_config(config)
            logger.info(f"RAG Pipeline initialized from {config_path} (attempt {attempt})")
            break  # success — exit the retry loop

        except Exception as e:
            logger.warning(
                f"Pipeline init attempt {attempt}/{max_attempts} failed: {e}"
            )
            if attempt < max_attempts:
                import asyncio
                logger.info(f"Retrying in {retry_delay}s ...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "Pipeline failed to initialize after all retry attempts. "
                    "Check Milvus logs: docker compose logs milvus",
                    exc_info=True,
                )

    yield  # Application runs here

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


# ── Debug Request / Response schemas ───────────────────────────────────────────
class DebugChunksRequest(BaseModel):
    directory_path: str
    save_to_file: Optional[bool] = False
    output_format: Optional[str] = "json"  # json or txt


class ChunkInfo(BaseModel):
    id: int
    doc_id: str
    chunk_index: int
    text: str
    breadcrumb: str
    page_number: int
    filename: str
    char_count: int
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DebugChunksResponse(BaseModel):
    chunks: List[ChunkInfo]
    total_chunks: int
    processing_time_ms: float


class DebugRetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    metadata_filter: Optional[Dict[str, Any]] = None


class RetrievedDocInfo(BaseModel):
    text: str
    doc_id: str
    chunk_index: int
    rrf_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    breadcrumb: str
    page_number: int


class DebugRetrieveResponse(BaseModel):
    query: str
    retrieved_docs: List[RetrievedDocInfo]
    total_candidates: int
    retrieval_time_ms: float


class DebugRerankRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    rerank_top_k: Optional[int] = 5


class RerankedDocInfo(BaseModel):
    text: str
    doc_id: str
    chunk_index: int
    rrf_score: float
    rerank_score: float
    final_score: float
    breadcrumb: str
    page_number: int


class DebugRerankResponse(BaseModel):
    query: str
    reranked_docs: List[RerankedDocInfo]
    rerank_time_ms: float


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


@app.post("/debug/chunks", response_model=DebugChunksResponse, summary="Debug: View chunks before embedding")
async def debug_chunks(request: DebugChunksRequest):
    """
    View chunks after chunking but before embedding.
    Useful for inspecting how documents are being split and processed.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    import time
    start_time = time.time()

    try:
        # Process the directory to get chunks
        chunks = rag_pipeline.ingestion.process_directory(request.directory_path)

        # Prepare chunk data for response
        chunk_data = []
        for i, chunk in enumerate(chunks):
            # Estimate token count (rough approximation: 1 token ≈ 4 chars)
            token_count = len(chunk.text) // 4
            
            chunk_info = ChunkInfo(
                id=i,
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                breadcrumb=chunk.breadcrumb,
                page_number=chunk.page_number,
                filename=chunk.filename,
                char_count=len(chunk.text),
                token_count=token_count,
                metadata=dict(chunk.metadata) if chunk.metadata else {}
            )
            chunk_data.append(chunk_info)

        # Optionally save to file
        if request.save_to_file:
            import json
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunks_debug_{timestamp}.{request.output_format}"
            filepath = f"./debug_output/{filename}"
            
            # Create debug_output directory if it doesn't exist
            import os
            os.makedirs("./debug_output", exist_ok=True)
            
            if request.output_format == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump([chunk.dict() for chunk in chunk_data], f, indent=2, ensure_ascii=False)
            else:  # txt format
                with open(filepath, 'w', encoding='utf-8') as f:
                    for chunk in chunk_data:
                        f.write(f"=== CHUNK {chunk.id} ===\n")
                        f.write(f"Doc ID: {chunk.doc_id}\n")
                        f.write(f"Breadcrumb: {chunk.breadcrumb}\n")
                        f.write(f"Page: {chunk.page_number}\n")
                        f.write(f"Filename: {chunk.filename}\n")
                        f.write(f"Text: {chunk.text}\n")
                        f.write("\n" + "="*50 + "\n\n")

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return DebugChunksResponse(
            chunks=chunk_data,
            total_chunks=len(chunk_data),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Debug chunks failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/retrieve", response_model=DebugRetrieveResponse, summary="Debug: View retrieval results before reranking")
async def debug_retrieve(request: DebugRetrieveRequest):
    """
    View retrieval results after RRF fusion but before reranking.
    Shows how documents are retrieved and scored by the hybrid system.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    import time
    start_time = time.time()

    try:
        # Perform retrieval only (without reranking)
        docs = rag_pipeline.retriever.retrieve(
            query=request.query,
            collection_name=rag_pipeline.config.storage.collection_name,
            metadata_filter=request.metadata_filter,
            k=request.k,
        )

        # Prepare retrieved docs data for response
        retrieved_docs = []
        for doc in docs:
            # Note: We don't have individual dense/sparse scores here since
            # the retriever returns the fused RRF results. We'll leave them as None.
            doc_info = RetrievedDocInfo(
                text=doc.text,
                doc_id=doc.doc_id,
                chunk_index=doc.chunk_index,
                rrf_score=doc.score,
                dense_score=None,
                sparse_score=None,
                breadcrumb=doc.metadata.get("breadcrumb", ""),
                page_number=doc.metadata.get("page_number") or 0
            )
            retrieved_docs.append(doc_info)

        retrieval_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return DebugRetrieveResponse(
            query=request.query,
            retrieved_docs=retrieved_docs,
            total_candidates=len(retrieved_docs),
            retrieval_time_ms=retrieval_time
        )
    except Exception as e:
        logger.error(f"Debug retrieve failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/rerank", response_model=DebugRerankResponse, summary="Debug: View rerank results before LLM")
async def debug_rerank(request: DebugRerankRequest):
    """
    View reranking results after Jina reranking but before LLM generation.
    Shows how the reranker modifies the order and scores of retrieved documents.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    import time
    start_time = time.time()

    try:
        # Get the top k documents before reranking
        docs = rag_pipeline.retriever.retrieve(
            query=request.query,
            collection_name=rag_pipeline.config.storage.collection_name,
            k=request.k,
        )

        # Apply reranking using the retriever's internal method
        reranked_docs = rag_pipeline.retriever._rerank(request.query, docs)

        # Take only the top rerank_top_k documents
        reranked_docs = reranked_docs[:request.rerank_top_k]

        # Prepare reranked docs data for response
        reranked_docs_info = []
        for doc in reranked_docs:
            doc_info = RerankedDocInfo(
                text=doc.text,
                doc_id=doc.doc_id,
                chunk_index=doc.chunk_index,
                rrf_score=doc.score,  # This is the RRF score before reranking
                rerank_score=doc.score,  # The score after reranking (this is actually the reranked score)
                final_score=doc.score,  # Same as rerank_score in this context
                breadcrumb=doc.metadata.get("breadcrumb", ""),
                page_number=doc.metadata.get("page_number") or 0
            )
            reranked_docs_info.append(doc_info)

        rerank_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return DebugRerankResponse(
            query=request.query,
            reranked_docs=reranked_docs_info,
            rerank_time_ms=rerank_time
        )
    except Exception as e:
        logger.error(f"Debug rerank failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
