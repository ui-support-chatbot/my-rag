from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path

from config import (
    RAGConfig,
    IngestionConfig,
    EmbeddingConfig,
    StorageConfig,
    RetrievalConfig,
    GenerationConfig,
    EvaluationConfig,
)
from ingestion import IngestionPipeline, PDFParser, HTMLParser, Chunker
from embedding import DenseEmbeddingModel, SparseEmbeddingModel
from storage import MilvusClient
from retrieval import Retriever
from generation import LLM, DEFAULT_SYSTEM_PROMPT
from evaluation import RAGASEvaluator
from evaluation.synthetic_qa import SyntheticQAGenerator
from debugging import ChunkInspector, RetrievalTracer

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    context: str
    retrieved_docs: List[Any]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Main RAG pipeline combining all components."""

    def __init__(
        self,
        config: RAGConfig,
        ingestion: IngestionPipeline = None,
        dense_model: DenseEmbeddingModel = None,
        sparse_model: SparseEmbeddingModel = None,
        storage: MilvusClient = None,
        retriever: Retriever = None,
        llm: LLM = None,
    ):
        self.config = config
        self.ingestion = ingestion or IngestionPipeline(
            chunk_size=config.ingestion.chunk_size,
            chunk_overlap=config.ingestion.chunk_overlap,
            embedding_model=config.embedding.dense_model,
        )
        self.dense_model = dense_model or DenseEmbeddingModel(
            model_name=config.embedding.dense_model,
            device=config.embedding.device,
        )
        self.sparse_model = sparse_model or SparseEmbeddingModel(
            model_name=config.embedding.sparse_model,
            device=config.embedding.device,
        )
        self.storage = storage or MilvusClient(
            uri=config.storage.milvus_uri,
            db_name=config.storage.db_name,
        )
        self.retriever = retriever or Retriever(
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            milvus_client=self.storage,
            reranker_model=config.retrieval.reranker_model,
            k=config.retrieval.k,
            hybrid_weight=config.retrieval.hybrid_weight,
        )
        self.llm = llm or LLM(
            endpoint=config.generation.llm_endpoint,
            model_name=config.generation.model_name,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
        )
        self.evaluator = None
        self.tracer = RetrievalTracer(self.retriever)

    @classmethod
    def from_config(cls, config: RAGConfig):
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str):
        config = RAGConfig.from_yaml(path)
        return cls(config)

    def ingest(
        self,
        paths: List[str] = None,
        directory: str = None,
        doc_id_prefix: str = "doc",
    ) -> int:
        """Ingest documents into the vector store."""
        collection_name = self.config.storage.collection_name
        self.storage.create_collection(collection_name, self.dense_model.dimension)

        chunks = []
        if paths:
            for i, path in enumerate(paths):
                doc_id = f"{doc_id_prefix}_{i:03d}"
                file_chunks = self.ingestion.process_file(path, doc_id=doc_id)
                chunks.extend(file_chunks)

        if directory:
            dir_chunks = self.ingestion.process_directory(directory)
            chunks.extend(dir_chunks)

        data = []
        for chunk in chunks:
            # Use embed_documents (no query prompt, full neural model for sparse).
            # embed_query is reserved for user queries at retrieval time.
            dense_emb = self.dense_model.embed_documents([chunk.text])[0]
            sparse_emb = self.sparse_model.embed_documents([chunk.text])[0]
            data.append(
                {
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "breadcrumb": chunk.breadcrumb,
                    "page_number": chunk.page_number,
                    "source": chunk.filename,
                    **chunk.metadata,
                }
            )

        if data:
            self.storage.insert(collection_name, data)
            logger.info(f"Indexed {len(data)} chunks")

        return len(data)

    def query(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
        k: Optional[int] = None,
        return_context: bool = True,
    ) -> RAGResult:
        """Query the RAG system."""
        docs = self.retriever.retrieve(
            query=query,
            collection_name=self.config.storage.collection_name,
            doc_ids=doc_ids,
            metadata_filter=metadata_filter,
            k=k or self.config.retrieval.k,
        )

        # LLM.generate now handles context formatting with breadcrumbs
        result = self.llm.generate(
            prompt=query,
            retrieved_docs=docs,
            context=None if docs else "No context provided.",
        )

        return RAGResult(
            answer=result.answer,
            context=result.context,
            retrieved_docs=docs,
            sources=result.sources,
            metadata={"query": query, "num_docs": len(docs)},
        )

    def query_with_keyword_check(
        self,
        query: str,
        check_keyword: str,
        doc_ids: Optional[List[str]] = None,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Query with keyword verification in retrieved docs."""
        trace = self.tracer.trace_retrieve(
            query=query,
            doc_ids=doc_ids,
            k=k,
            check_keyword=check_keyword,
        )

        docs = trace["documents"]
        answer = self.llm.generate(
            query, context="\n\n".join([d["text_preview"] for d in docs])
        ).answer

        return {
            "query": query,
            "check_keyword": check_keyword,
            "answer": answer,
            "retrieval_trace": trace,
        }

    def find_keyword(
        self, keyword: str, doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find all chunks containing a keyword."""
        inspector = ChunkInspector(self.storage, self.config.storage.collection_name)
        matches = inspector.find_chunks_with_keyword(keyword, doc_id)

        return [
            {
                "chunk_id": m.chunk_id,
                "doc_id": m.doc_id,
                "chunk_index": m.chunk_index,
                "text_preview": m.chunk_text[:200] + "..."
                if len(m.chunk_text) > 200
                else m.chunk_text,
                "keyword_positions": m.keyword_positions,
            }
            for m in matches
        ]

    def evaluate(
        self,
        questions: List[str],
        synthetic_qa: bool = False,
        ground_truths: Optional[List[str]] = None,
        retrieval_logs: Optional[List[Dict]] = None,
        rerank_logs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the RAG pipeline."""
        if self.evaluator is None:
            self.evaluator = RAGASEvaluator(
                eval_llm=self.llm.client,
                metrics=self.config.evaluation.metrics,
            )

        contexts = []
        answers = []

        for q in questions:
            result = self.query(q)
            contexts.append([result.context])
            answers.append(result.answer)

        eval_results = self.evaluator.evaluate(
            questions=questions,
            contexts=contexts,
            answers=answers,
            ground_truths=ground_truths,
            retrieval_logs=retrieval_logs,
            rerank_logs=rerank_logs,
        )

        return eval_results

    def generate_synthetic_qa(
        self,
        paths: List[str] = None,
        directory: str = None,
        num_qa_per_doc: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate synthetic Q&A pairs from documents."""
        if not paths and not directory:
            raise ValueError("Must provide paths or directory")

        if paths:
            chunks = []
            for i, path in enumerate(paths):
                file_chunks = self.ingestion.process_file(path, doc_id=f"doc_{i}")
                chunks.extend([c.text for c in file_chunks])
        else:
            dir_chunks = self.ingestion.process_directory(directory)
            chunks = [c.text for c in dir_chunks]

        generator = SyntheticQAGenerator(self.llm)
        qa_pairs = generator.generate(chunks, num_qa_per_doc)

        return qa_pairs
