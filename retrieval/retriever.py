from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    text: str
    doc_id: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Retriever:
    """Retriever with metadata filtering and reranking support."""

    def __init__(
        self,
        dense_model,
        sparse_model,
        milvus_client,
        reranker_model: Optional[str] = "jinaai/jina-reranker-v3",
        k: int = 50,  # Increased candidate pool
        hybrid_weight: float = 0.5,
    ):
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.milvus = milvus_client
        self.reranker_model = reranker_model
        self.k = k
        self.hybrid_weight = hybrid_weight
        self._reranker = None
        self.device = "cuda" if _has_cuda() else "cpu"

    @property
    def reranker(self):
        """Lazily load jina-reranker-v3 via AutoModel with trust_remote_code.

        The model exposes a high-level .rerank(query, docs) method and does NOT
        use the standard sequence-classification head, so we must NOT load it
        with AutoModelForSequenceClassification.
        """
        if self._reranker is None and self.reranker_model:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self.reranker_model,
                dtype="auto",
                trust_remote_code=True,
            )
            model.eval()
            if _has_cuda():
                model = model.cuda()
            self._reranker = {"model": model}
        return self._reranker

    def _build_filter(
        self,
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Optional[str]:
        if not doc_ids and not metadata_filter:
            return None

        conditions = []
        if doc_ids:
            ids_str = ",".join(f'"{d}"' for d in doc_ids)
            conditions.append(f"doc_id in [{ids_str}]")

        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')

        return " and ".join(conditions)

    def _apply_rrf(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Custom Reciprocal Rank Fusion (RRF)
        score(d) = sum(1 / (k + rank_i))
        """
        scores = {}

        # Dense ranks
        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sparse ranks
        for rank, res in enumerate(sparse_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Map IDs back to full documents
        all_docs = {res["id"]: res for res in dense_results + sparse_results}

        results = []
        for doc_id in sorted_ids:
            res = all_docs[doc_id]
            entity = res["entity"]
            results.append(
                RetrievedDocument(
                    text=entity["text"],
                    doc_id=entity["doc_id"],
                    chunk_index=entity["chunk_index"],
                    score=scores[doc_id],
                    metadata={
                        "source": entity.get("source"),
                        "breadcrumb": entity.get("breadcrumb"),
                        "page_number": entity.get("page_number"),
                    },
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        collection_name: str = "documents",
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
        k: Optional[int] = None,
        search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
    ) -> List[RetrievedDocument]:
        candidate_k = k or self.k
        if candidate_k == 0:
            return []

        dense_emb = [self.dense_model.embed_query(query)]
        sparse_emb = [self.sparse_model.embed_query(query)]

        filter_expr = self._build_filter(doc_ids, metadata_filter)

        search_param_1 = {
            "data": dense_emb,
            "anns_field": "dense_embedding",
            "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
            "limit": candidate_k,
            "expr": filter_expr,
        }

        search_param_2 = {
            "data": [sparse_emb[0]],
            "anns_field": "sparse_embedding",
            "param": {"metric_type": "IP", "params": {"nprobe": 10}},
            "limit": candidate_k,
            "expr": filter_expr,
        }

        from pymilvus import AnnSearchRequest

        req_1 = AnnSearchRequest(**search_param_1)
        req_2 = AnnSearchRequest(**search_param_2)

        # Perform separate searches for RRF
        dense_results = self.milvus.client.search(
            collection_name=collection_name,
            data=dense_emb,
            anns_field="dense_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=candidate_k,
            expr=filter_expr,
            output_fields=[
                "text",
                "doc_id",
                "chunk_index",
                "source",
                "breadcrumb",
                "page_number",
            ],
        )[0]

        sparse_results = self.milvus.client.search(
            collection_name=collection_name,
            data=sparse_emb[0],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=candidate_k,
            expr=filter_expr,
            output_fields=[
                "text",
                "doc_id",
                "chunk_index",
                "source",
                "breadcrumb",
                "page_number",
            ],
        )[0]

        # Apply RRF Fusion (k=60)
        docs = self._apply_rrf(dense_results, sparse_results, k=60)

        # Log Top-50 candidates BEFORE reranking
        logger.info(f"Retrieved {len(docs)} candidates after RRF fusion")
        for i, doc in enumerate(docs[:10]):
            logger.debug(
                f"Rank {i + 1}: Score={doc.score:.4f} | Doc={doc.doc_id} | Text={doc.text[:50]}..."
            )

        if self.reranker and docs:
            docs = self._rerank(query, docs)

        return docs

    def _rerank(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank using jina-reranker-v3's built-in .rerank() method.

        The model accepts (query, [doc_texts]) and returns a list of dicts:
            [{"document": str, "relevance_score": float, "index": int}, ...]
        sorted from most to least relevant.
        """
        if not docs or not self.reranker:
            return docs

        model = self.reranker["model"]
        doc_texts = [d.text for d in docs]

        results = model.rerank(query, doc_texts)

        # results is already sorted by relevance_score descending.
        # Map back to original RetrievedDocument objects by original index.
        logger.info("Reranking complete.")
        reranked = []
        for i, res in enumerate(results):
            original_doc = docs[res["index"]]
            logger.info(
                f"Reranked Rank {i + 1}: Score={res['relevance_score']:.4f} | Doc={original_doc.doc_id}"
            )
            reranked.append(original_doc)

        return reranked

    def find_chunks_with_keyword(
        self,
        keyword: str,
        collection_name: str = "documents",
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> List[RetrievedDocument]:
        """Find chunks containing a specific keyword."""
        filter_expr = f'doc_id == "{doc_id}"' if doc_id else None

        results = self.milvus.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=[
                "text",
                "doc_id",
                "chunk_index",
                "source",
                "breadcrumb",
                "page_number",
            ],
            limit=10000,
        )

        matches = []
        for res in results:
            text = res["text"]
            search_text = text if case_sensitive else text.lower()
            search_keyword = keyword if case_sensitive else keyword.lower()

            if search_keyword in search_text:
                matches.append(
                    RetrievedDocument(
                        text=text,
                        doc_id=res["doc_id"],
                        chunk_index=res["chunk_index"],
                        score=1.0 if case_sensitive else 0.0,
                        metadata={
                            "source": res.get("source"),
                            "keyword": keyword,
                            "breadcrumb": res.get("breadcrumb"),
                            "page_number": res.get("page_number"),
                        },
                    )
                )

        return matches


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
