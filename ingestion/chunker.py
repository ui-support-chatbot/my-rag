from typing import List
from ingestion.base import ChunkRecord
import logging
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

logger = logging.getLogger(__name__)


class Chunker:
    """Hybrid chunker using Docling for token-aware, structure-aware splitting."""

    def __init__(
        self,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        from docling.chunking import HybridChunker

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model),
            max_tokens=chunk_size,
        )

        self.chunker = HybridChunker(
            tokenizer=tokenizer, merge_peers=True, repeat_table_header=True
        )

    def chunk(self, docling_doc, filename: str, doc_id: str = "") -> List[ChunkRecord]:
        """
        Use HybridChunker to preserve document structure and provide contextualized text.
        """
        chunks = []

        for chunk in self.chunker.chunk(docling_doc):
            # Contextualized text: prepends heading hierarchy and other structure
            contextualized_text = self.chunker.contextualize(chunk=chunk)

            # Breadcrumb: joined heading hierarchy
            breadcrumb = (
                " > ".join(chunk.meta.headings) if chunk.meta.headings else "Root"
            )

            # Page number
            page_number = None
            if chunk.meta.doc_items:
                item = chunk.meta.doc_items[0]
                if hasattr(item, "prov") and item.prov:
                    page_number = item.prov[0].page_no

            chunks.append(
                ChunkRecord(
                    text=contextualized_text,
                    doc_id=doc_id or chunk.chunk_id,
                    chunk_index=len(chunks),
                    breadcrumb=breadcrumb,
                    page_number=page_number,
                    filename=filename,
                    metadata=chunk.meta.to_dict()
                    if hasattr(chunk.meta, "to_dict")
                    else {},
                )
            )

        return chunks
