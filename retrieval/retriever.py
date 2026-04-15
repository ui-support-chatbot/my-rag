from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import logging

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
    """Hybrid retriever: dense (Harrier) + sparse (OpenSearch) → RRF → Jina reranker."""

    def __init__(
        self,
        dense_model,
        sparse_model,
        milvus_client,
        reranker_model: Optional[str] = "jinaai/jina-reranker-v3",
        k: int = 50,
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
                    conditions.append(f"{key} == {value}")

        return " and ".join(conditions)

    def _apply_rrf(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion (RRF):
            score(d) = Σ  1 / (k + rank_i)

        k=60 is the standard constant from the original RRF paper.
        """
        scores: Dict[str, float] = {}

        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, res in enumerate(sparse_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Map internal Milvus IDs back to full document payloads
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
                    score=scores[doc_id],  # RRF fusion score
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

        dense_emb = self.dense_model.embed_query(query)   # uses 'web_search_query' prompt
        sparse_emb = self.sparse_model.embed_query(query)  # IDF-table only, no GPU pass

        filter_expr = self._build_filter(doc_ids, metadata_filter)

        output_fields = ["text", "doc_id", "chunk_index", "source", "breadcrumb", "page_number"]
        search_kwargs = dict(limit=candidate_k, expr=filter_expr, output_fields=output_fields)

        # ── Dense search ──────────────────────────────────────────────────────
        dense_results = self.milvus.client.search(
            collection_name=collection_name,
            data=[dense_emb],
            anns_field="dense_embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            **search_kwargs,
        )[0]

        # ── Sparse search ─────────────────────────────────────────────────────
        sparse_results = self.milvus.client.search(
            collection_name=collection_name,
            data=[sparse_emb],
            anns_field="sparse_embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            **search_kwargs,
        )[0]

        # ── RRF Fusion (k=60, the standard constant) ──────────────────────────
        docs = self._apply_rrf(dense_results, sparse_results, k=60)

        logger.info(f"Retrieved {len(docs)} candidates after RRF fusion")
        for i, doc in enumerate(docs[:10]):
            logger.debug(
                f"RRF Rank {i + 1}: score={doc.score:.5f} | "
                f"doc={doc.doc_id} | text={doc.text[:60]}..."
            )

        # ── Reranking ─────────────────────────────────────────────────────────
        if self.reranker and docs:
            docs = self._rerank(query, docs)

        return docs

    def _rerank(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank using jina-reranker-v3's built-in .rerank() method.

        The model returns a list of dicts:
            [{"document": str, "relevance_score": float, "index": int}, ...]
        sorted from most to least relevant.

        We update each document's `.score` with the reranker's relevance_score
        so that downstream consumers get the correct confidence value.
        """
        if not docs or not self.reranker:
            return docs

        model = self.reranker["model"]
        doc_texts = [d.text for d in docs]

        results = model.rerank(query, doc_texts)

        reranked = []
        for i, res in enumerate(results):
            original_doc = docs[res["index"]]
            # ✅ Update score to the reranker's relevance_score (was RRF score before)
            original_doc.score = float(res["relevance_score"])
            logger.info(
                f"Reranked #{i + 1}: score={original_doc.score:.4f} | doc={original_doc.doc_id}"
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
            output_fields=["text", "doc_id", "chunk_index", "source", "breadcrumb", "page_number"],
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
                        score=1.0,
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
