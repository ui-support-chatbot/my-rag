from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    context: str
    sources: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


class LLM:
    """Local LLM wrapper supporting vLLM and HuggingFace."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1",
        model_name: str = "llama-3-8b",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(base_url=self.endpoint, api_key="dummy")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        return self._client

    def generate(
        self,
        prompt: str,
        retrieved_docs: Optional[List[Any]] = None,
        context: Optional[str] = None,
    ) -> GenerationResult:
        from generation.prompts import DEFAULT_SYSTEM_PROMPT

        # Format context with Source [breadcrumb] markers
        if retrieved_docs:
            formatted_context = "\n\n".join(
                [
                    f"Source [{doc.metadata.get('breadcrumb', 'Unknown')}]: {doc.text}"
                    for doc in retrieved_docs
                ]
            )
        else:
            formatted_context = context or "No context provided."

        system_content = DEFAULT_SYSTEM_PROMPT.format(context=formatted_context)

        user_content = prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = response.choices[0].message.content

            sources = []
            if retrieved_docs:
                sources = [
                    {
                        "breadcrumb": doc.metadata.get("breadcrumb", "Unknown"),
                        "filename": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page_number", "Unknown"),
                    }
                    for doc in retrieved_docs
                ]

            return GenerationResult(
                answer=answer, context=formatted_context, sources=sources
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return GenerationResult(
                answer="Error generating response",
                context=formatted_context,
                sources=[],
            )

    def batch_generate(
        self, prompts: List[str], contexts: Optional[List[str]] = None
    ) -> List[GenerationResult]:
        if contexts is None:
            contexts = [None] * len(prompts)

        results = []
        for prompt, ctx in zip(prompts, contexts):
            results.append(self.generate(prompt, context=ctx))
        return results
