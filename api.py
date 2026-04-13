from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from pipeline import RAGPipeline
from config import RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="My RAG System API",
    description="API for high-performance structure-aware RAG using Harrier, OpenSearch, and Jina.",
    version="0.1.0",
)

# Global pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

class QueryRequest(BaseModel):
    query: str
    config_override: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    directory_path: str

class RAGResponse(BaseModel):
    answer: str
    context: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    config_path = os.getenv("RAG_CONFIG_PATH", "config_rag.yaml")
    try:
        if os.path.exists(config_path):
            config = RAGConfig.from_yaml(config_path)
            rag_pipeline = RAGPipeline.from_config(config)
            logger.info(f"RAG Pipeline initialized from {config_path}")
        else:
            logger.warning(f"Config file {config_path} not found. Pipeline not initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy" if rag_pipeline else "uninitialized",
        "milvus": "connected" if rag_pipeline and rag_pipeline.storage else "not_connected"
    }
    return status

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        result = rag_pipeline.query(request.query)
        return {
            "answer": result.answer,
            "context": result.context,
            "sources": result.sources,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def background_ingestion(directory: str):
    if not rag_pipeline:
        logger.error("Ingestion failed: Pipeline not initialized")
        return
    try:
        logger.info(f"Starting background ingestion for {directory}")
        rag_pipeline.ingest(directory)
        logger.info(f"Ingestion completed for {directory}")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")

@app.post("/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not os.path.exists(request.directory_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
    
    background_tasks.add_task(background_ingestion, request.directory_path)
    return {"message": f"Ingestion of {request.directory_path} started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
