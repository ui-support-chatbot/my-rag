from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Enhanced chunk record for structure-aware RAG."""

    text: str
    doc_id: str
    chunk_index: int
    breadcrumb: str = ""
    page_number: Optional[int] = None
    filename: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseParser(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def accepts_extension(ext: str) -> bool:
        pass
