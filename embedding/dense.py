from typing import List
import numpy as np
from embedding.base import BaseEmbeddingModel


class DenseEmbeddingModel(BaseEmbeddingModel):
    """Dense embeddings using sentence-transformers.

    Supports instruction-based models like microsoft/harrier-oss-v1-0.6b that
    use a decoder-only architecture with last-token pooling. Queries are encoded
    with a task-specific prompt (prompt_name) while documents are encoded plainly.
    """

    def __init__(
        self,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        device: str = "cuda",
        query_prompt_name: str = "web_search_query",
        quantize_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        # prompt_name used for query encoding (instruction-tuned models need this).
        # Set to None to disable for standard bi-encoders like BGE.
        self.query_prompt_name = query_prompt_name
        self.quantize_8bit = quantize_8bit
        self._model = None
        self._dim = None

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"dtype": "auto", "trust_remote_code": True}
            if self.quantize_8bit:
                model_kwargs["load_in_8bit"] = True

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
            )
        return self._model

    def unload(self):
        if self._model is not None:
            import gc
            import torch
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def model(self):
        return self.load()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            model = self.load()
            # get_sentence_embedding_dimension() was renamed in sentence-transformers 3.x
            if hasattr(model, "get_embedding_dimension"):
                self._dim = model.get_embedding_dimension()
            else:
                self._dim = model.get_sentence_embedding_dimension()
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode documents — no instruction prompt needed."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encode a query, applying the task prompt if configured."""
        kwargs = {}
        if self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        return self.model.encode([text], convert_to_numpy=True, **kwargs)[0].tolist()
