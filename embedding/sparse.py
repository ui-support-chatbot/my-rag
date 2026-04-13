from typing import List, Dict
import torch
from embedding.base import BaseEmbeddingModel


class SparseEmbeddingModel(BaseEmbeddingModel):
    """Sparse embeddings using the OpenSearch neural sparse encoder.

    Model: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte

    This is an ASYMMETRIC / INFERENCE-FREE model:
    - Documents: run through the full neural model (encode_document).
    - Queries:   use ONLY the tokenizer + a pre-computed IDF weight lookup table
                 (encode_query). This makes query-time extremely fast.

    Requires: pip install -U sentence-transformers
    The SparseEncoder API (sentence_transformers >= 3.x) handles IDF loading and
    the asymmetric encode automatically.
    """

    def __init__(
        self,
        model_name: str = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers.sparse_encoder import SparseEncoder

            self._model = SparseEncoder(
                self.model_name,
                trust_remote_code=True,
                device=self.device,
                # Pin the revision used in the official example for reproducibility.
                model_kwargs={"code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"},
            )
        return self._model

    @property
    def dimension(self) -> int:
        # Vocabulary size of the underlying BERT-like tokenizer (30_522 tokens).
        return 30522

    def _tensor_to_dict(self, sparse_tensor) -> Dict[int, float]:
        """Convert a sparse tensor (1-D or COO) to a {token_id: weight} dict
        as expected by Milvus sparse vector fields."""
        if hasattr(sparse_tensor, "to_sparse"):
            # Dense tensor returned by some sentence-transformer versions
            sparse_tensor = sparse_tensor.to_sparse()

        if sparse_tensor.is_sparse:
            indices = sparse_tensor.coalesce().indices().squeeze(0).tolist()
            values = sparse_tensor.coalesce().values().tolist()
            return {int(i): float(v) for i, v in zip(indices, values) if float(v) > 0}

        # Already a plain dict/mapping
        if isinstance(sparse_tensor, dict):
            return {int(k): float(v) for k, v in sparse_tensor.items() if float(v) > 0}

        # Fallback: dense 1-D tensor
        nonzero_idx = sparse_tensor.nonzero(as_tuple=True)[0].tolist()
        return {int(i): float(sparse_tensor[i].item()) for i in nonzero_idx}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """Encode documents using the full neural model."""
        embeddings = self.model.encode_documents(texts)
        if not isinstance(embeddings, (list, tuple)):
            # Batch tensor — split per sample
            embeddings = [embeddings[i] for i in range(len(texts))]
        return [self._tensor_to_dict(e) for e in embeddings]

    def embed_query(self, text: str) -> Dict[int, float]:
        """Encode a query using the IDF table only (no model forward pass)."""
        embedding = self.model.encode_queries(text)
        # encode_queries may return a single tensor or a list
        if isinstance(embedding, (list, tuple)):
            embedding = embedding[0]
        return self._tensor_to_dict(embedding)
