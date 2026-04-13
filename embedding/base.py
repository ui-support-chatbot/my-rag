from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @property
    def dimension(self) -> int:
        raise NotImplementedError
