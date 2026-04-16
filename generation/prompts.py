DEFAULT_SYSTEM_PROMPT = """You are a precise assistant. Answer ONLY using the provided context. 

If the answer is not present in the context, say explicitly: "I don't have enough information to answer that question."

Cite sources using the breadcrumb provided in the context (e.g., [www.ui.ac.id > Team]).
If a URL is provided for a source, you may mention it if the user asks for more information or if it helps verify the answer.

Context:
{context}
"""

DEFAULT_USER_PROMPT = """{question}"""

RAGAS_EVALUATION_PROMPT = """You are an expert evaluator assessing a RAG system's output.

Evaluate the following:
- Question: {question}
- Retrieved Context: {context}
- Generated Answer: {answer}

Provide scores (0-1) for:
1. Faithfulness: Does the answer stay true to the context?
2. Answer Relevance: Does the answer address the question?
3. Context Precision: Are the most relevant chunks ranked highest?
4. Context Recall: Does the context contain information needed to answer?

Output as JSON with keys: faithfulness, answer_relevancy, context_precision, context_recall
"""

SYNTHETIC_QA_PROMPT = """Based on the following document, generate {num_qa} diverse question-answer pairs that test different aspects of the content.

Include:
- Factoid questions (who, what, when, where)
- How/why questions (explanations)
- Comparison questions

Document:
{doc}

Output as JSON array:
[{{"question": "...", "answer": "..."}}]
"""
