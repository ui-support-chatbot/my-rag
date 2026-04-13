from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMatch:
    chunk_id: str
    chunk_text: str
    doc_id: str
    chunk_index: int
    keyword_positions: List[int]
    score: float = 1.0
