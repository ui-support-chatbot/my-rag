from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RetrievalTracer:
    """Trace retrieval process for debugging."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.trace_log = []

    def trace_retrieve(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        k: int = 5,
        check_keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trace retrieval with optional keyword verification."""
        docs = self.retriever.retrieve(
            query=query,
            doc_ids=doc_ids,
            k=k,
        )

        trace = {
            "query": query,
            "doc_ids_filter": doc_ids,
            "k": k,
            "num_retrieved": len(docs),
            "documents": [],
        }

        for i, doc in enumerate(docs):
            doc_info = {
                "rank": i + 1,
                "doc_id": doc.doc_id,
                "chunk_index": doc.chunk_index,
                "score": doc.score,
                "text_preview": doc.text[:200] + "..."
                if len(doc.text) > 200
                else doc.text,
            }

            if check_keyword:
                keyword_present = check_keyword.lower() in doc.text.lower()
                doc_info["contains_keyword"] = keyword_present

            trace["documents"].append(doc_info)

        self.trace_log.append(trace)
        return trace

    def get_trace_log(self) -> List[Dict[str, Any]]:
        return self.trace_log
