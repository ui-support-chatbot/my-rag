from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import json
import os
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
from ingestion.state import IngestionState

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    context: str
    retrieved_docs: List[Any]
    sources: List[Dict[str, Any]]
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionRunSummary:
    indexed_chunks: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)


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
            chunking_strategy=config.ingestion.chunking_strategy,
            pdf_parser=config.ingestion.pdf_parser,
            html_parser=config.ingestion.html_parser,
        )
        self.dense_model = dense_model or DenseEmbeddingModel(
            model_name=config.embedding.dense_model,
            device=config.embedding.dense_device,
            quantize_8bit=config.embedding.quantize_8bit,
        )
        self.sparse_model = sparse_model or SparseEmbeddingModel(
            model_name=config.embedding.sparse_model,
            device=config.embedding.sparse_device,
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
            reranker_endpoint=config.retrieval.reranker_endpoint,
            k=config.retrieval.k,
            hybrid_weight=config.retrieval.hybrid_weight,
        )
        self.retriever.load_models()
        self.llm = llm or LLM(
            endpoint=config.generation.llm_endpoint,
            model_name=config.generation.model_name,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
        )
        self.evaluator = None
        self.tracer = RetrievalTracer(self.retriever)
        self.ingestion_state = IngestionState(
            state_path=config.ingestion.state_path
        )

    @classmethod
    def from_config(cls, config: RAGConfig) -> "RAGPipeline":
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str) -> "RAGPipeline":
        config = RAGConfig.from_yaml(path)
        return cls(config)

    def ingest(
        self,
        paths: List[str] = None,
        directory: str = None,
        doc_id_prefix: str = "doc",
    ) -> int:
        """Ingest documents into the vector store using batch embedding (Incremental)."""
        collection_name = self.config.storage.collection_name
        self.storage.create_collection(collection_name, self.dense_model.dimension)
        return self._ingest_resilient(paths, directory, doc_id_prefix, collection_name)


    def _ingest_resilient(
        self,
        paths: List[str],
        directory: str,
        doc_id_prefix: str,
        collection_name: str,
    ) -> int:
        all_file_paths = []
        if paths:
            all_file_paths.extend(paths)
        if directory:
            if os.path.isfile(directory):
                all_file_paths.append(directory)
            else:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in [".pdf", ".html", ".htm"]):
                            all_file_paths.append(os.path.join(root, file))

        summary = IngestionRunSummary()
        files_to_process = []
        if self.config.ingestion.incremental:
            for path in all_file_paths:
                status = self.ingestion_state.get_file_status(path)
                if status == "unchanged":
                    logger.info(f"Skipping {path}: No changes detected (Incremental)")
                    summary.skipped_files += 1
                else:
                    files_to_process.append((path, status))
        else:
            files_to_process = [(path, "new") for path in all_file_paths]

        if not files_to_process:
            logger.info("Nothing to ingest. Everything is up to date.")
            return 0

        batch_size = max(1, self.config.embedding.batch_size)
        total_files = len(files_to_process)
        logger.info(
            f"Starting ingestion: {total_files} file(s), "
            f"batch_size={batch_size}, skipped={summary.skipped_files}"
        )

        for index, (path, status) in enumerate(files_to_process):
            doc_id = f"{doc_id_prefix}_{index:03d}"
            try:
                logger.info(f"[{index + 1}/{total_files}] Processing {path} ({status})")
                file_chunks = self.ingestion.process_file(path, doc_id=doc_id)
                if not file_chunks:
                    raise ValueError("No chunks produced")

                if self.config.ingestion.save_snapshots:
                    self.save_chunks_before_embedding(
                        chunks=file_chunks,
                        output_prefix=f"ingest_{Path(path).stem}",
                    )

                records = self._embed_chunks_with_retry(file_chunks, batch_size)

                # Delete old or partial rows only after replacement vectors are ready.
                self.storage.delete_by_source(collection_name, path)
                self.storage.insert(collection_name, records)
                self.ingestion_state.update_file(
                    file_path=path,
                    doc_id=doc_id,
                    chunk_count=len(records),
                )

                summary.indexed_chunks += len(records)
                summary.successful_files += 1
                logger.info(f"Indexed {len(records)} chunk(s) from {path}")
            except Exception as e:
                summary.failed_files += 1
                summary.failures.append({"path": path, "error": str(e)})
                logger.error(f"Failed to ingest {path}: {e}", exc_info=True)
                self._clear_cuda_cache()

        logger.info(
            "Ingestion summary: "
            f"indexed_chunks={summary.indexed_chunks}, "
            f"successful_files={summary.successful_files}, "
            f"failed_files={summary.failed_files}, "
            f"skipped_files={summary.skipped_files}"
        )
        if summary.failures:
            logger.warning(f"Ingestion failures: {summary.failures}")
        logger.info("Ingestion complete. Keeping embedding models loaded.")
        return summary.indexed_chunks

    def _embed_chunks_with_retry(self, chunks: List[Any], batch_size: int) -> List[Dict[str, Any]]:
        records = []
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in chunk_batch]
            dense_embeddings = self._embed_batch_with_retry(
                self.dense_model.embed_documents,
                texts,
                batch_size=len(texts),
                label="dense",
            )
            sparse_embeddings = self._embed_batch_with_retry(
                self.sparse_model.embed_documents,
                texts,
                batch_size=len(texts),
                label="sparse",
            )
            if len(dense_embeddings) != len(chunk_batch) or len(sparse_embeddings) != len(chunk_batch):
                raise ValueError(
                    "Embedding count mismatch: "
                    f"chunks={len(chunk_batch)}, dense={len(dense_embeddings)}, "
                    f"sparse={len(sparse_embeddings)}"
                )
            for chunk, dense_emb, sparse_emb in zip(chunk_batch, dense_embeddings, sparse_embeddings):
                records.append(self._build_storage_record(chunk, dense_emb, sparse_emb))
        return records

    def _embed_batch_with_retry(self, embed_fn, texts: List[str], batch_size: int, label: str) -> List[Any]:
        try:
            embeddings = []
            for start in range(0, len(texts), batch_size):
                embeddings.extend(embed_fn(texts[start : start + batch_size]))
            return embeddings
        except Exception as e:
            self._clear_cuda_cache()
            if batch_size <= 1:
                raise
            smaller_batch = max(1, batch_size // 2)
            logger.warning(
                f"{label} embedding failed for {len(texts)} text(s) at "
                f"batch_size={batch_size}: {e}. Retrying with batch_size={smaller_batch}."
            )
            return self._embed_batch_with_retry(embed_fn, texts, smaller_batch, label)

    def _build_storage_record(self, chunk: Any, dense_emb: Any, sparse_emb: Any) -> Dict[str, Any]:
        source_path = os.path.abspath(chunk.filename)
        record = {
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "dense_embedding": dense_emb,
            "sparse_embedding": sparse_emb,
            "pdf_url": chunk.metadata.get("pdf_url"),
            "page_url": chunk.metadata.get("page_url"),
            "scraped_at": chunk.metadata.get("scraped_at"),
            "page_number": chunk.page_number,
        }
        record.update(chunk.metadata)
        record["source"] = source_path
        record["filename"] = chunk.filename
        return record

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to clear CUDA cache", exc_info=True)

    def save_chunks_before_embedding(
        self,
        paths: List[str] = None,
        directory: str = None,
        chunks: List[Any] = None,
        output_prefix: str = "manual",
    ) -> List:
        """
        Saves a snapshot of chunks to disk for debugging/inspection.
        Can process files/directories OR accept a list of pre-processed chunks.
        """
        if not chunks:
            chunks = []
            if paths:
                for i, path in enumerate(paths):
                    file_chunks = self.ingestion.process_file(path, doc_id=f"doc_{i:03d}")
                    chunks.extend(file_chunks)
            if directory:
                dir_chunks = self.ingestion.process_directory(directory)
                chunks.extend(dir_chunks)

        if not chunks:
            logger.warning("No chunks to save.")
            return []

        # Serialization logic
        from datetime import datetime
        import os

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_prefix}_{timestamp}.json"
        snapshot_dir = Path("storage/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filepath = snapshot_dir / filename

        chunk_data = [
            {
                "id": i,
                "doc_id": c.doc_id,
                "text": c.text,
                "breadcrumb": c.breadcrumb,
                "token_estimate": len(c.text) // 4,
                "metadata": dict(c.metadata) if c.metadata else {},
            }
            for i, c in enumerate(chunks)
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunk snapshot saved to {filepath}")
        return chunks

    def query(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
        k: Optional[int] = None,
        return_context: bool = True,
        pre_retrieved_docs: Optional[List[Any]] = None,
    ) -> RAGResult:
        """
        Run the full RAG pipeline:
          1. Embed query (dense + sparse)
          2. Milvus hybrid search -> RRF fusion (top-k candidates)
          3. Jina-v3 reranking
          4. Slice to rerank_top_k for LLM context window
          5. Calculate confidence score
          6. Grounded generation
        """
        if pre_retrieved_docs:
            docs = pre_retrieved_docs
        else:
            docs = self.retriever.retrieve(
                query=query,
                collection_name=self.config.storage.collection_name,
                doc_ids=doc_ids,
                metadata_filter=metadata_filter,
                k=k or self.config.retrieval.k,
            )
        # Slice to rerank_top_k to avoid LLM token overflow.
        # After reranking, docs are sorted best-first; we take only the top N.
        rerank_top_k = self.config.retrieval.rerank_top_k
        docs = docs[:rerank_top_k]
        logger.info(
            f"Passing top-{rerank_top_k} reranked docs to LLM "
            f"(retrieved {len(docs)} total after RRF)"
        )

        # Confidence scoring (LLM-based)
        confidence_score = self.llm.get_confidence_score(query, docs)

        # LLM.generate handles context formatting with Source [breadcrumb] markers
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
            confidence_score=confidence_score,
            metadata={"query": query, "num_docs": len(docs)},
        )

    def query_stream(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
        k: Optional[int] = None,
        pre_retrieved_docs: Optional[List[Any]] = None,
        confidence_score: Optional[float] = None,
    ):
        """Streaming version of the RAG pipeline query."""

        try:
            if pre_retrieved_docs:
                docs = pre_retrieved_docs
            else:
                docs = self.retriever.retrieve(
                    query=query,
                    collection_name=self.config.storage.collection_name,
                    doc_ids=doc_ids,
                    metadata_filter=metadata_filter,
                    k=k or self.config.retrieval.k,
                )
                rerank_top_k = self.config.retrieval.rerank_top_k
                docs = docs[:rerank_top_k]

            # If confidence wasn't provided (e.g. called directly from pipeline, not API)
            # calculate it now.
            if confidence_score is None:
                confidence_score = self.llm.get_confidence_score(query, docs)

            # Yield search metadata (Confidence + Doc count)
            metadata_payload = {
                "confidence_score": confidence_score,
                "num_docs": len(docs),
                "query": query
            }
            logger.info(f"Streaming metadata: {metadata_payload}")
            yield f"data: {json.dumps({'type': 'metadata', 'content': metadata_payload})}\n\n"

            # Yield sources separately for backward compatibility/UI richness
            sources = [
                {
                    "pdf_url": doc.metadata.get("pdf_url"),
                    "page_url": doc.metadata.get("page_url"),
                    "scraped_at": doc.metadata.get("scraped_at"),
                    "page": doc.metadata.get("page_number", "Unknown"),
                }
                for doc in docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            # Stream the LLM response
            for token in self.llm.generate_stream(prompt=query, retrieved_docs=docs):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        except Exception as e:
            logger.error(f"Streaming query failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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
                "text_preview": (
                    m.chunk_text[:200] + "..." if len(m.chunk_text) > 200 else m.chunk_text
                ),
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
        """Evaluate the RAG pipeline using RAGAS metrics."""
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
