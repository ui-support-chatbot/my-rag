from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IngestionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "fixed"
    pdf_parser: str = "docling"
    html_parser: str = "trafilatura"
    save_snapshots: bool = False
    incremental: bool = True
    state_path: str = "storage/ingestion_state.json"
    upload_dir: str = "uploads"


@dataclass
class EmbeddingConfig:
    dense_model: str = "microsoft/harrier-oss-v1-0.6b"
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
    device: str = "cuda"  # Multi-GPU: This is the default for Dense
    dense_device: str = "cuda:0"
    sparse_device: str = "cpu"  # Sparse query is fast on CPU
    quantize_8bit: bool = True
    batch_size: int = 32


@dataclass
class StorageConfig:
    milvus_uri: str = "./milvus.db"
    collection_name: str = "documents"
    db_name: str = "default"
    metric_type: str = "COSINE"


@dataclass
class RetrievalConfig:
    k: int = 50
    """Candidate pool size fetched from Milvus before reranking."""

    rerank_top_k: int = 5
    """Number of documents passed to the LLM after reranking."""

    hybrid_weight: float = 0.5
    reranker_model: Optional[str] = "jinaai/jina-reranker-v3"
    reranker_quantize_8bit: bool = False
    reranker_device: str = "cuda:1"  # Put reranker on second GPU
    min_score: float = 0.0


@dataclass
class GenerationConfig:
    llm_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "llama-3-8b"
    max_tokens: int = 512
    temperature: float = 0.0
    system_prompt: str = (
        "Use the following context to answer the question. "
        "If the context does not contain enough information to answer, say so.\n\n"
        "Context:\n{context}"
    )


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    eval_llm: str = "gpt-4o-mini"
    eval_embeddings: str = "all-MiniLM-L6-v2"
    num_synthetic_qa: int = 3


@dataclass
class RAGConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Load config from a YAML file. Any missing top-level key falls back to defaults."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            ingestion=IngestionConfig(**data.get("ingestion", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            storage=StorageConfig(**data.get("storage", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )
