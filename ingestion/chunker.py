from typing import List
from ingestion.base import ChunkRecord
import logging
import os
logger = logging.getLogger(__name__)


class Chunker:
    """Chunk Docling documents and plain extracted text."""

    def __init__(
        self,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
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
        self.chunker = None

    def chunk(
        self,
        docling_doc,
        filename: str,
        doc_id: str = "",
        external_metadata: dict = None,
    ) -> List[ChunkRecord]:
        """
        Use HierarchicalChunker to preserve document structure.
        """
        from docling.chunking import HierarchicalChunker

        if external_metadata is None:
            external_metadata = {}
        if self.chunker is None:
            self.chunker = HierarchicalChunker()

        chunks = []
        # Priority: pdf_url > source_url > page_url
        source_url = (
            external_metadata.get("pdf_url")
            or external_metadata.get("source_url")
            or external_metadata.get("page_url", "")
        )

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

    def chunk_text(
        self,
        text: str,
        filename: str,
        doc_id: str = "",
        external_metadata: dict = None,
    ) -> List[ChunkRecord]:
        """Split plain text into overlapping chunks using configured token-like units."""
        if external_metadata is None:
            external_metadata = {}

        normalized_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        if not normalized_text:
            return []

        source_url = (
            external_metadata.get("pdf_url")
            or external_metadata.get("source_url")
            or external_metadata.get("page_url", "")
        )

        records = []
        for chunk_text in self._split_text(normalized_text):
            metadata = dict(external_metadata)
            metadata["source_url"] = source_url
            metadata["chunking_strategy"] = "standard_text"

            records.append(
                ChunkRecord(
                    text=chunk_text,
                    doc_id=doc_id or os.path.splitext(os.path.basename(filename))[0],
                    chunk_index=len(records),
                    breadcrumb="",
                    page_number=None,
                    filename=filename,
                    metadata=metadata,
                )
            )

        return records

    def _split_text(self, text: str) -> List[str]:
        return self._split_tokens(text.split())

    def _split_tokens(self, tokens: List[str]) -> List[str]:
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + self.chunk_size]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks
