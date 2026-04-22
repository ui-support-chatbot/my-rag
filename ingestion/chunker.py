from typing import List
from ingestion.base import ChunkRecord
import logging
import os
logger = logging.getLogger(__name__)


class Chunker:
    """Hierarchical chunker using Docling structure-aware splitting."""

    def __init__(
        self,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        from docling.chunking import HierarchicalChunker

        if chunk_overlap:
            logger.debug(
                "chunk_overlap=%s is ignored by Docling HierarchicalChunker.",
                chunk_overlap,
            )
        logger.info(
            "Using Docling HierarchicalChunker. chunk_size=%s is retained for "
            "config visibility, but splitting follows document structure.",
            chunk_size,
        )
        self.chunker = HierarchicalChunker()

    def chunk(self, docling_doc, filename: str, doc_id: str = "", external_metadata: dict = None) -> List[ChunkRecord]:
        """
        Use HierarchicalChunker to preserve document structure.
        """
        if external_metadata is None:
            external_metadata = {}

        chunks = []
        
        # Extract source signals for the breadcrumb
        domain = external_metadata.get("domain", "")
        # Priority: pdf_url > source_url > page_url
        source_url = external_metadata.get("pdf_url") or external_metadata.get("source_url") or external_metadata.get("page_url", "")

        for chunk in self.chunker.chunk(docling_doc):
            chunk_text = chunk.text

            # --- Breadcrumb replacement ---
            # We no longer use structural breadcrumbs. Metadata URLs are used instead downstream.
            breadcrumb = ""

            # Page number
            page_number = None
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                item = chunk.meta.doc_items[0]
                if hasattr(item, "prov") and item.prov:
                    page_number = item.prov[0].page_no

            # Merge Docling metadata with external metadata
            full_metadata = chunk.meta.to_dict() if hasattr(chunk.meta, "to_dict") else {}
            full_metadata.update(external_metadata)
            # Ensure source_url is explicitly present
            full_metadata["source_url"] = source_url

            chunks.append(
                ChunkRecord(
                    text=chunk_text,
                    doc_id=doc_id or chunk.chunk_id,
                    chunk_index=len(chunks),
                    breadcrumb=breadcrumb,
                    page_number=page_number,
                    filename=filename,
                    metadata=full_metadata,
                )
            )

        return chunks
