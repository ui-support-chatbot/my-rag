from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChunkInspector:
    """Inspect chunks and trace keyword presence for debugging."""

    def __init__(self, milvus_client, collection_name: str = "documents"):
        self.milvus = milvus_client
        self.collection_name = collection_name

    def find_chunks_with_keyword(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> List["ChunkMatch"]:
        """Find all chunks containing the keyword."""
        from debugging.inspector import ChunkMatch

        filter_expr = None
        if doc_id:
            filter_expr = f'doc_id == "{doc_id}"'

        results = self.milvus.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id", "text", "doc_id", "chunk_index", "source"],
            limit=10000,
        )

        matches = []
        for res in results:
            text = res.get("text", "")
            search_text = text if case_sensitive else text.lower()
            search_keyword = keyword if case_sensitive else keyword.lower()

            if search_keyword in search_text:
                positions = self._find_positions(text, keyword, case_sensitive)
                matches.append(
                    ChunkMatch(
                        chunk_id=res.get("id", ""),
                        chunk_text=text,
                        doc_id=res.get("doc_id", ""),
                        chunk_index=res.get("chunk_index", 0),
                        keyword_positions=positions,
                    )
                )

        logger.info(
            f"Found {len(matches)} chunks containing '{keyword}'"
            + (f" in doc {doc_id}" if doc_id else "")
        )
        return matches

    def _find_positions(
        self, text: str, keyword: str, case_sensitive: bool
    ) -> List[int]:
        """Find character positions of keyword in text."""
        positions = []
        search_text = text if case_sensitive else text.lower()
        search_keyword = keyword if case_sensitive else keyword.lower()

        start = 0
        while True:
            pos = search_text.find(search_keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def get_doc_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        results = self.milvus.query(
            collection_name=self.collection_name,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["id", "text", "doc_id", "chunk_index", "source"],
            limit=10000,
        )
        return results

    def inspect_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific chunk."""
        results = self.milvus.query(
            collection_name=self.collection_name,
            filter=f'id == "{chunk_id}"',
            output_fields=["id", "text", "doc_id", "chunk_index", "source"],
            limit=1,
        )
        return results[0] if results else None
