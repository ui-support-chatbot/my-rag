from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import json
import os
import uuid
from datetime import datetime
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
from generation import LLM
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
    duplicate_files: int = 0
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
            pdf_parser=config.ingestion.pdf_parser,
            pdf_chunking_strategy=config.ingestion.pdf_chunking_strategy,
            html_parser=config.ingestion.html_parser,
            html_chunking_strategy=config.ingestion.html_chunking_strategy,
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
            reasoning_effort=config.generation.reasoning_effort,
            system_prompt=config.generation.system_prompt,
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
        all_file_paths = self._discover_ingestion_files(paths, directory)

        summary = IngestionRunSummary()
        run_id = uuid.uuid4().hex
        started_at = datetime.now()
        job_snapshot = self._new_ingestion_snapshot(
            run_id=run_id,
            started_at=started_at,
            input_paths=paths,
            directory=directory,
            discovered_files=all_file_paths,
        )
        files_to_process = []
        pending_job_duplicates = []
        seen_hashes: Dict[str, Dict[str, Any]] = {}

        if self.config.ingestion.incremental:
            for path in all_file_paths:
                fingerprint = self.ingestion_state.scan_file(path)
                classification = self.ingestion_state.classify_fingerprint(fingerprint)
                status = classification["status"]
                canonical = classification.get("canonical")
                if fingerprint.hash in seen_hashes and status in {"new", "modified"}:
                    status = "duplicate"
                    classification["status"] = status
                    classification["reason"] = "same content as another file in this job"
                    classification["canonical_job_entry"] = seen_hashes[fingerprint.hash]
                else:
                    seen_hashes.setdefault(
                        fingerprint.hash,
                        {
                            "path": fingerprint.path,
                            "doc_id": canonical.doc_id if canonical else self._doc_id_for_fingerprint(fingerprint.hash, doc_id_prefix),
                        },
                    )

                if status == "unchanged":
                    logger.info(f"Skipping {path}: No changes detected (incremental)")
                    summary.skipped_files += 1
                    job_snapshot["files"].append(
                        self._snapshot_manifest_entry(
                            path=path,
                            fingerprint=fingerprint,
                            status=status,
                            reason=classification["reason"],
                            doc_id=classification["canonical"].doc_id,
                            chunk_count=classification["canonical"].chunk_count,
                        )
                    )
                elif status == "duplicate":
                    summary.duplicate_files += 1
                    if classification.get("canonical"):
                        canonical_doc = self._record_duplicate_alias(
                            collection_name=collection_name,
                            path=path,
                            fingerprint=fingerprint,
                            classification=classification,
                        )
                        job_snapshot["files"].append(
                            self._snapshot_manifest_entry(
                                path=path,
                                fingerprint=fingerprint,
                                status=status,
                                reason=classification["reason"],
                                doc_id=canonical_doc["doc_id"],
                                chunk_count=0,
                                canonical_path=canonical_doc["path"],
                                canonical_doc_id=canonical_doc["doc_id"],
                            )
                        )
                    else:
                        pending_job_duplicates.append((path, fingerprint, classification))
                else:
                    files_to_process.append((path, status, fingerprint))
        else:
            for path in all_file_paths:
                fingerprint = self.ingestion_state.scan_file(path)
                classification = self.ingestion_state.classify_fingerprint(fingerprint)
                status = classification["status"]
                canonical = classification.get("canonical")
                if fingerprint.hash in seen_hashes and status in {"new", "modified", "unchanged"}:
                    status = "duplicate"
                    classification["status"] = status
                    classification["reason"] = "same content as another file in this job"
                    classification["canonical_job_entry"] = seen_hashes[fingerprint.hash]
                else:
                    seen_hashes.setdefault(
                        fingerprint.hash,
                        {
                            "path": fingerprint.path,
                            "doc_id": canonical.doc_id if canonical else self._doc_id_for_fingerprint(fingerprint.hash, doc_id_prefix),
                        },
                    )

                if status == "duplicate":
                    summary.duplicate_files += 1
                    if classification.get("canonical"):
                        canonical_doc = self._record_duplicate_alias(
                            collection_name=collection_name,
                            path=path,
                            fingerprint=fingerprint,
                            classification=classification,
                        )
                        job_snapshot["files"].append(
                            self._snapshot_manifest_entry(
                                path=path,
                                fingerprint=fingerprint,
                                status=status,
                                reason=classification["reason"],
                                doc_id=canonical_doc["doc_id"],
                                chunk_count=0,
                                canonical_path=canonical_doc["path"],
                                canonical_doc_id=canonical_doc["doc_id"],
                            )
                        )
                    else:
                        pending_job_duplicates.append((path, fingerprint, classification))
                else:
                    files_to_process.append((path, "modified" if status == "unchanged" else status, fingerprint))

        if not files_to_process:
            logger.info("Nothing to ingest. Everything is up to date.")
            self._finalize_ingestion_snapshot(job_snapshot, summary, started_at)
            return 0

        batch_size = max(1, self.config.embedding.batch_size)
        total_files = len(files_to_process)
        logger.info(
            f"Starting ingestion: {total_files} file(s), "
            f"batch_size={batch_size}, skipped={summary.skipped_files}, "
            f"duplicates={summary.duplicate_files}"
        )

        for index, (path, status, fingerprint) in enumerate(files_to_process):
            doc_id = self._doc_id_for_fingerprint(fingerprint.hash, doc_id_prefix)
            try:
                logger.info(f"[{index + 1}/{total_files}] Processing {path} ({status})")
                file_chunks = self.ingestion.process_file(path, doc_id=doc_id)
                if not file_chunks:
                    raise ValueError("No chunks produced")

                records = self._embed_chunks_with_retry(file_chunks, batch_size)

                # Delete old or partial rows only after replacement vectors are ready.
                self.storage.delete_by_source(collection_name, path)
                self.storage.insert(collection_name, records)
                self.ingestion_state.update_file(
                    file_path=path,
                    doc_id=doc_id,
                    chunk_count=len(records),
                    fingerprint=fingerprint,
                    metadata={"status": "canonical"},
                )

                summary.indexed_chunks += len(records)
                summary.successful_files += 1
                job_snapshot["files"].append(
                    self._snapshot_manifest_entry(
                        path=path,
                        fingerprint=fingerprint,
                        status=status,
                        reason="processed and embedded",
                        doc_id=doc_id,
                        chunk_count=len(records),
                        chunks=file_chunks,
                    )
                )
                pending_job_duplicates = self._finalize_pending_job_duplicates(
                    pending_job_duplicates=pending_job_duplicates,
                    content_hash=fingerprint.hash,
                    canonical_doc={"path": fingerprint.path, "doc_id": doc_id},
                    collection_name=collection_name,
                    job_snapshot=job_snapshot,
                )
                logger.info(f"Indexed {len(records)} chunk(s) from {path}")
            except Exception as e:
                summary.failed_files += 1
                summary.failures.append({"path": path, "error": str(e)})
                job_snapshot["files"].append(
                    self._snapshot_manifest_entry(
                        path=path,
                        fingerprint=fingerprint,
                        status="failed",
                        reason=str(e),
                        doc_id=doc_id,
                        chunk_count=0,
                    )
                )
                logger.error(f"Failed to ingest {path}: {e}", exc_info=True)
                self._clear_cuda_cache()

        for path, fingerprint, classification in pending_job_duplicates:
            summary.failed_files += 1
            summary.failures.append(
                {
                    "path": path,
                    "error": "Duplicate canonical file was not successfully indexed",
                }
            )
            job_snapshot["files"].append(
                self._snapshot_manifest_entry(
                    path=path,
                    fingerprint=fingerprint,
                    status="failed",
                    reason="duplicate canonical file was not successfully indexed",
                    doc_id=classification.get("canonical_job_entry", {}).get("doc_id", ""),
                    chunk_count=0,
                    canonical_path=classification.get("canonical_job_entry", {}).get("path"),
                    canonical_doc_id=classification.get("canonical_job_entry", {}).get("doc_id"),
                )
            )

        logger.info(
            "Ingestion summary: "
            f"indexed_chunks={summary.indexed_chunks}, "
            f"successful_files={summary.successful_files}, "
            f"failed_files={summary.failed_files}, "
            f"skipped_files={summary.skipped_files}, "
            f"duplicate_files={summary.duplicate_files}"
        )
        if summary.failures:
            logger.warning(f"Ingestion failures: {summary.failures}")
        self._finalize_ingestion_snapshot(job_snapshot, summary, started_at)
        logger.info("Ingestion complete. Keeping embedding models loaded.")
        return summary.indexed_chunks

    def _finalize_pending_job_duplicates(
        self,
        pending_job_duplicates: List[Any],
        content_hash: str,
        canonical_doc: Dict[str, str],
        collection_name: str,
        job_snapshot: Dict[str, Any],
    ) -> List[Any]:
        remaining = []
        for path, fingerprint, classification in pending_job_duplicates:
            if fingerprint.hash != content_hash:
                remaining.append((path, fingerprint, classification))
                continue

            canonical_doc_for_alias = self._record_duplicate_alias(
                collection_name=collection_name,
                path=path,
                fingerprint=fingerprint,
                classification={
                    **classification,
                    "canonical_job_entry": canonical_doc,
                },
            )
            job_snapshot["files"].append(
                self._snapshot_manifest_entry(
                    path=path,
                    fingerprint=fingerprint,
                    status="duplicate",
                    reason=classification.get("reason", "same content as another file in this job"),
                    doc_id=canonical_doc_for_alias["doc_id"],
                    chunk_count=0,
                    canonical_path=canonical_doc_for_alias["path"],
                    canonical_doc_id=canonical_doc_for_alias["doc_id"],
                )
            )
        return remaining

    def _discover_ingestion_files(self, paths: List[str], directory: str) -> List[str]:
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
        return sorted(dict.fromkeys(all_file_paths))

    def _doc_id_for_fingerprint(self, content_hash: str, doc_id_prefix: str) -> str:
        prefix = doc_id_prefix or "doc"
        return f"{prefix}_{content_hash[:12]}"

    def _record_duplicate_alias(
        self,
        collection_name: str,
        path: str,
        fingerprint: Any,
        classification: Dict[str, Any],
    ) -> Dict[str, str]:
        canonical = classification.get("canonical")
        canonical_job_entry = classification.get("canonical_job_entry")
        if canonical:
            canonical_doc = {"path": canonical.path, "doc_id": canonical.doc_id}
        elif canonical_job_entry:
            canonical_doc = {
                "path": canonical_job_entry["path"],
                "doc_id": canonical_job_entry["doc_id"],
            }
        else:
            raise ValueError(f"Duplicate classification for {path} has no canonical file")

        existing = classification.get("existing")
        if existing and existing.path == fingerprint.path and existing.doc_id != canonical_doc["doc_id"]:
            self.storage.delete_by_source(collection_name, path)

        from ingestion.state import FileState

        canonical_state = canonical or FileState(
            path=canonical_doc["path"],
            hash=fingerprint.hash,
            last_modified=fingerprint.last_modified,
            doc_id=canonical_doc["doc_id"],
        )
        self.ingestion_state.record_alias(
            file_path=path,
            canonical=canonical_state,
            fingerprint=fingerprint,
            metadata={"reason": classification.get("reason", "duplicate content")},
        )
        logger.info(
            f"Skipping duplicate {fingerprint.path}; canonical={canonical_doc['path']}"
        )
        return canonical_doc

    def _new_ingestion_snapshot(
        self,
        run_id: str,
        started_at: datetime,
        input_paths: List[str],
        directory: str,
        discovered_files: List[str],
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "inputs": {
                "paths": input_paths or [],
                "directory": directory,
            },
            "config": {
                "incremental": self.config.ingestion.incremental,
                "pdf_parser": self.config.ingestion.pdf_parser,
                "pdf_chunking_strategy": self.config.ingestion.pdf_chunking_strategy,
                "html_parser": self.config.ingestion.html_parser,
                "html_chunking_strategy": self.config.ingestion.html_chunking_strategy,
                "chunk_size": self.config.ingestion.chunk_size,
                "chunk_overlap": self.config.ingestion.chunk_overlap,
                "dense_model": self.config.embedding.dense_model,
                "sparse_model": self.config.embedding.sparse_model,
                "collection_name": self.config.storage.collection_name,
            },
            "discovered_file_count": len(discovered_files),
            "files": [],
            "totals": {},
        }

    def _snapshot_manifest_entry(
        self,
        path: str,
        fingerprint: Any,
        status: str,
        reason: str,
        doc_id: str,
        chunk_count: int,
        canonical_path: str = None,
        canonical_doc_id: str = None,
        chunks: List[Any] = None,
    ) -> Dict[str, Any]:
        entry = {
            "path": os.path.abspath(path),
            "status": status,
            "reason": reason,
            "content_hash": fingerprint.hash,
            "size": fingerprint.size,
            "last_modified": fingerprint.last_modified,
            "doc_id": doc_id,
            "chunk_count": chunk_count,
        }
        if canonical_path:
            entry["canonical_path"] = canonical_path
        if canonical_doc_id:
            entry["canonical_doc_id"] = canonical_doc_id
        if chunks is not None:
            entry["chunks"] = [
                {
                    "chunk_index": c.chunk_index,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "breadcrumb": c.breadcrumb,
                    "page_number": c.page_number,
                    "token_estimate": len(c.text) // 4,
                    "metadata": dict(c.metadata) if c.metadata else {},
                }
                for c in chunks
            ]
        return entry

    def _finalize_ingestion_snapshot(
        self,
        snapshot: Dict[str, Any],
        summary: IngestionRunSummary,
        started_at: datetime,
    ) -> None:
        snapshot["finished_at"] = datetime.now().isoformat()
        snapshot["duration_seconds"] = (
            datetime.fromisoformat(snapshot["finished_at"]) - started_at
        ).total_seconds()
        snapshot["totals"] = {
            "indexed_chunks": summary.indexed_chunks,
            "successful_files": summary.successful_files,
            "failed_files": summary.failed_files,
            "skipped_files": summary.skipped_files,
            "duplicate_files": summary.duplicate_files,
            "manifest_entries": len(snapshot["files"]),
        }
        snapshot["failures"] = summary.failures

        if not self.config.ingestion.save_snapshots:
            return

        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path("storage/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filepath = snapshot_dir / f"ingest_job_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Ingestion job snapshot saved to {filepath}")

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
        retrieved_count = len(docs)
        docs = docs[:rerank_top_k]
        stage_label = "reranked" if self.retriever.reranker else "RRF-ranked (No dedicated reranker)"
        logger.info(
            f"Passing top-{len(docs)} {stage_label} docs to LLM "
            f"(from {retrieved_count} retrieved candidates)"
        )

        result = self.llm.generate(
            prompt=query,
            retrieved_docs=docs,
            context=None if docs else "No context provided.",
        )

        confidence_score = (
            result.confidence_score
            if result.confidence_score is not None
            else self.llm.get_confidence_score(query, docs)
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
