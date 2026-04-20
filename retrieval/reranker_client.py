import json
import logging
from typing import List, Dict, Any, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


class LlamaServerReranker:
    """HTTP client for llama.cpp's reranking endpoint.

    The server is expected to run a GGUF reranker model with reranking enabled.
    See llama.cpp docs for the /reranking, /rerank, /v1/rerank aliases.
    """

    def __init__(self, endpoint: str, model: Optional[str] = None, timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        if not self.endpoint:
            return []
        if not documents:
            return []

        payload: Dict[str, Any] = {
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        if self.model:
            payload["model"] = self.model

        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            logger.error("Reranker request failed: %s", exc)
            raise

        data = json.loads(raw)
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            results = data.get("results", data)
        else:
            results = []

        normalized: List[Dict[str, Any]] = []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                score = item.get("relevance_score", item.get("score"))
                if index is None or score is None:
                    continue
                normalized.append(
                    {
                        "index": int(index),
                        "relevance_score": float(score),
                    }
                )

        normalized.sort(key=lambda item: item["relevance_score"], reverse=True)
        return normalized[: len(documents)]
