# MMORE: The TL;DR Guide

Welcome to **MMORE** (Massive Multimodal Open RAG & Extraction). If you're wondering what everything does without reading hundreds of lines of code, this is the guide for you!

---

## 🚀 1. What does MMORE actually do?
MMORE is an **end-to-end framework** designed to make AI chatbots smart about your private documents. It automates 4 major steps:
1. **Process (Extraction):** Takes messy PDFs, Word Docs, or raw HTML (like the Beasiswa website) and perfectly converts them into structured Markdown text and images.
2. **Post-Process (Chunking):** Slices those massive documents into smaller "chunks" (like paragraph by paragraph) so the AI can digest them easily.
3. **Index (Embedding):** Converts those chunks into numbers (Vector Embeddings) and saves them into a Vector Database (like **Milvus**). 
4. **RAG (Retrieval-Augmented Generation):** Hosts an API you can ask questions to. It searches the database for the right chunks, hands them to an LLM (Language Model), and generates a human-like answer based *strictly* on your documents.

---

## 🧠 2. The Models: What is available?
MMORE is "model agnostic"—it doesn't force you into one ecosystem. You can plug and play depending on your budget and server hardware.

* **LLMs (The Brains that answer you):** 
  * Proprietary: OpenAI (`GPT-4o`), Anthropic (`Claude`), Mistral, Cohere.
  * Local/Open Source: HuggingFace models, or **Ollama** (`Qwen`, `Llama 3`). You use Ollama when you want it to be 100% free and private on your own server.
* **Embedders (The Translators that convert text to searchable vectors):**
  * Dense Models: `all-MiniLM-L6-v2` (fast, lightweight), `BAAI/bge-m3` (great for Indonesian/multilingual).
  * Sparse Models: `SPLADE` (excellent for exact-keyword matching).

---

## ⚖️ 3. Why do we need a "Reranker"?
In your configs, you might have noticed a `reranker_model_name: BAAI/bge-reranker-base`. 

**Why choose a reranker?**
When you ask a question, the Vector Database quickly fetches the "Top 5" chunks that are mathematically similar to your question. However, sometimes chunks have similar keywords but *don't actually answer the question*. 
A Reranker acts as a highly intelligent judge. It takes those 5 fast results, quickly reads them against your exact question, and re-orders them so the absolute most relevant chunk is forced to the #1 spot before the LLM sees it. **It massively reduces AI hallucinations.**

---

## 🧪 4. Evaluation: How do we know it's good?
MMORE comes with a built-in `evaluator.py` that uses an industry-standard framework called **RAGAS** (Retrieval Augmented Generation Assessment). Instead of you reading 1,000 chatbot answers to see if they are correct, RAGAS uses a "Judge LLM" (like GPT-4) to grade the pipeline automatically from `0.0` to `1.0`.

### What kind of evaluation is there?
It grades two main things: **Did it find the right document?** and **Did it generate a good answer?**

1. **Faithfulness:** Did the AI hallucinate? (If it said the Beasiswa requires a 3.5 GPA, is that *actually* stated in the retrieved chunk, or did the AI make it up?)
2. **Answer Relevancy:** Did the AI actually answer your specific question, or did it dodge it and talk about something else?
3. **Context Precision / Recall:** Did the Vector database actually pull the correct documents from the thousands of PDFs available, or did it pull garbage?
4. **Factual Correctness & Noise Sensitivity:** Is the AI easily confused by irrelevant information injected into the prompt?

By running an evaluation, you get a scientific excel sheet showing exactly where your RAG pipeline fails, so you know whether you need a better Embedder, a better LLM, or better chunking!

### Can we do it automatically?
**YES!** You do not need a human tester to read the answers. MMORE totally automates this testing process via its `RAGEvaluator` code.

Here is how the automated test works behind the scenes:
1. **The Test Dataset:** You provide a spreadsheet or HuggingFace Dataset filled with dozens of "dummy" questions (e.g., "What is the GPA requirement?") and the "ground truth" right answer.
2. **The Run:** MMORE's evaluator automatically fires those questions at your RAG API. The vector database pulls the chunks, and your actual LLM (like Qwen) generates answers.
3. **The Judge:** An external "Judge LLM" (usually a highly accurate model like GPT-4o) reads both the "ground truth" and what your Qwen AI just generated. 
4. **The Scorecard:** The Judge LLM automatically calculates math scores (0.0 to 1.0) for Faithfulness, Relevancy, etc. and outputs a Pandas DataFrame table without you lifting a finger!
