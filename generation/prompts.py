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
[{"question": "...", "answer": "..."}]
"""

CONFIDENCE_CHECK_PROMPT = """You are a precise classifier.
Task: Rate the probability (0.0 to 1.0) that the provided context contains enough information to answer the query.

Calibration Guidelines:
- 1.0: Context contains the exact answer.
- 0.8: Answer is clearly present but requires minor connection of facts.
- 0.5: Context is on-topic but missing key details for a full answer.
- 0.2: Context mentions related keywords but lacks substance.
- 0.0: Context is completely irrelevant.

Rules:
- Respond with ANY decimal between 0.0 and 1.0. 
- Use the full range (e.g., 0.95, 0.72, 0.40).
- Output ONLY the score in this format: [SCORE: X.X]
- NO other text or explanation.

Query: {query}
Context: {context}
Result: """
