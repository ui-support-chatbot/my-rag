from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

METRIC_DESCRIPTIONS = {
    "faithfulness": {
        "name": "Faithfulness",
        "description": "Measures whether the answer stays true to the retrieved context. The LLM extracts claims from the answer and checks if each claim is supported by the context.",
        "inputs": "question, context, answer",
        "score_range": "0-1",
    },
    "answer_relevancy": {
        "name": "Answer Relevancy",
        "description": "Measures how well the answer addresses the original question. Generated answer is embedded and compared to the question.",
        "inputs": "question, answer",
        "score_range": "0-1",
    },
    "context_precision": {
        "name": "Context Precision",
        "description": "Measures whether the most relevant chunks are ranked highest in retrieval. LLM identifies relevant chunks and calculates precision.",
        "inputs": "question, contexts",
        "score_range": "0-1",
    },
    "context_recall": {
        "name": "Context Recall",
        "description": "Measures if retrieved context contains the information needed to answer. Uses ground truth or LLM-extracted facts.",
        "inputs": "question, context, ground_truth",
        "score_range": "0-1",
    },
}


class RAGASEvaluator:
    """RAGAS evaluation wrapper with detailed failure analysis."""

    def __init__(
        self, eval_llm=None, eval_embeddings=None, metrics: Optional[List[str]] = None
    ):
        self.eval_llm = eval_llm
        self.eval_embeddings = eval_embeddings
        self.metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    @staticmethod
    def explain_metrics() -> str:
        """Return human-readable explanation of RAGAS metrics."""
        lines = ["# RAGAS Metrics Explanation\n"]
        for key, info in METRIC_DESCRIPTIONS.items():
            lines.append(f"## {info['name']}")
            lines.append(f"- **What it measures**: {info['description']}")
            lines.append(f"- **Inputs**: {info['inputs']}")
            lines.append(f"- **Score range**: {info['score_range']}")
            lines.append("")
        return "\n".join(lines)

    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
        retrieval_logs: Optional[List[Dict]] = None,
        rerank_logs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation and perform failure categorization."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                Faithfulness,
                ResponseRelevancy,
                ContextPrecision,
                LLMContextRecall,
            )
            from ragas import EvaluationDataset
            from datasets import Dataset
        except ImportError:
            logger.warning("RAGAS not installed, returning mock results")
            return self._mock_evaluation(questions, contexts, answers)

        metric_map = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": ResponseRelevancy(),
            "context_precision": ContextPrecision(),
            "context_recall": LLMContextRecall(),
        }

        selected_metrics = [metric_map[m] for m in self.metrics if m in metric_map]

        dataset = Dataset.from_list(
            [
                {"question": q, "contexts": c, "answer": a, "ground_truth": g or ""}
                for q, c, a, g in zip(
                    questions,
                    contexts,
                    answers,
                    ground_truths or [None] * len(questions),
                )
            ]
        )

        eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

        try:
            result = evaluate(
                dataset=eval_dataset,
                metrics=selected_metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
            )
            metrics_df = result.to_pandas()

            # Failure Categorization
            analysis = self._categorize_failures(
                questions, metrics_df, retrieval_logs, rerank_logs
            )

            return {"metrics": metrics_df.to_dict(), "failure_analysis": analysis}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}

    def _categorize_failures(
        self, questions, metrics_df, retrieval_logs, rerank_logs
    ) -> List[Dict]:
        """Categorize failure cases for actionable insights."""
        failures = []

        for i in range(len(questions)):
            row = metrics_df.iloc[i]
            q = questions[i]

            # Logic for failure categorization
            # 1. Retrieval Failure: Context Recall is low
            if row.get("context_recall", 1.0) < 0.5:
                failures.append(
                    {
                        "query": q,
                        "type": "Retrieval Failure",
                        "reason": "The relevant information was not retrieved from the database.",
                    }
                )

            # 2. Reranking Failure: High recall but low precision/faithfulness
            # (i.e., info was in top-50 but not in top-5)
            elif row.get("context_precision", 1.0) < 0.5:
                failures.append(
                    {
                        "query": q,
                        "type": "Reranking Failure",
                        "reason": "Relevant information was retrieved but ranked too low for the LLM.",
                    }
                )

            # 3. Generation Failure: Context is good, but answer is wrong
            elif (
                row.get("faithfulness", 1.0) < 0.5
                or row.get("answer_relevancy", 1.0) < 0.5
            ):
                failures.append(
                    {
                        "query": q,
                        "type": "Generation Failure",
                        "reason": "LLM failed to synthesize a correct answer despite having the correct context.",
                    }
                )

        return failures

    def _mock_evaluation(self, questions, contexts, answers) -> Dict[str, Any]:
        """Return mock evaluation when RAGAS is not available."""
        return {
            "metrics": {
                "question": questions,
                "faithfulness": [0.0] * len(questions),
                "answer_relevancy": [0.0] * len(questions),
                "context_precision": [0.0] * len(questions),
                "context_recall": [0.0] * len(questions),
            },
            "failure_analysis": [],
        }
