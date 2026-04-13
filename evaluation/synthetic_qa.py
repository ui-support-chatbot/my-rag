from typing import List, Dict
import json
import logging

logger = logging.getLogger(__name__)


class SyntheticQAGenerator:
    """Generate synthetic Q&A pairs from documents for evaluation."""

    def __init__(self, llm):
        self.llm = llm

    def generate(
        self,
        documents: List[str],
        num_qa_per_doc: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs from documents."""
        qa_pairs = []

        for doc in documents:
            prompt = f"""Based on the following document, generate {num_qa_per_doc} diverse question-answer pairs.

Include different types:
- Factoid (who, what, when, where)
- How/why (explanations)
- Comparison questions

Document:
{doc[:2000]}

Output ONLY as JSON array:
[{{"question": "...", "answer": "..."}}]
"""
            result = self.llm.generate(prompt)

            try:
                json_start = result.answer.find("[")
                json_end = result.answer.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result.answer[json_start:json_end]
                    pairs = json.loads(json_str)
                    qa_pairs.extend(pairs)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Q&A JSON: {e}")

        return qa_pairs

    def generate_from_chunks(
        self,
        chunks: List[str],
        doc_ids: List[str],
        num_qa_per_chunk: int = 1,
    ) -> List[Dict[str, str]]:
        """Generate Q&A from individual chunks."""
        qa_pairs = []

        for chunk, doc_id in zip(chunks, doc_ids):
            if len(chunk.strip()) < 50:
                continue

            prompt = f"""Based on the following text chunk, generate {num_qa_per_chunk} question-answer pair(s).

Text:
{chunk[:1000]}

Output ONLY as JSON array:
[{{"question": "...", "answer": "..."}}]
"""
            result = self.llm.generate(prompt)

            try:
                json_start = result.answer.find("[")
                json_end = result.answer.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result.answer[json_start:json_end]
                    pairs = json.loads(json_str)
                    for p in pairs:
                        p["doc_id"] = doc_id
                    qa_pairs.extend(pairs)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Q&A JSON: {e}")

        return qa_pairs
