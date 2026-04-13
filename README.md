# MyRAG - Modular RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed for research and extensibility.

## Features

- **Advanced Ingestion**: PDF parsing via Docling, HTML extraction via Trafilatura, and Hybrid Chunking for token-aware, structure-preserving splitting.
- **Hybrid Search**: Combines Dense (BGE) and Sparse (SPLADE) embeddings using Milvus.
- **Reranking**: Integration with `bge-reranker-base` to improve retrieval precision.
- **Debuggability**: 
  - `find-keyword`: Locate specific keywords across all stored chunks.
  - `trace`: Verify if retrieved chunks contain specific keywords.
- **Evaluation**: Built-in RAGAS evaluation and synthetic QA generation.

## Installation

```bash
pip install -r src/my_rag/requirements.txt
```

## Usage

### 1. Configuration
Edit `config_rag.yaml` to set your LLM endpoint, embedding models, and Milvus URI.

### 2. Ingestion
```bash
python src/my_rag/cli.py ingest --config config_rag.yaml --directory ./data/docs
```

### 3. Querying
```bash
python src/my_rag/cli.py query --config config_rag.yaml --query "What is the main topic?"
```

### 4. Debugging
Find chunks containing a keyword:
```bash
python src/my_rag/cli.py find-keyword --config config_rag.yaml --keyword "machine learning"
```

Trace retrieval with a keyword check:
```bash
python src/my_rag/cli.py trace --config config_rag.yaml --query "..." --check-keyword "activation"
```

### 5. Evaluation
Generate synthetic QA and evaluate:
```bash
python src/my_rag/cli.py eval --config config_rag.yaml --synthetic --paths ./data/docs/sample.pdf
```

## Architecture

- `ingestion/`: Document parsing and chunking.
- `embedding/`: Dense and Sparse embedding model wrappers.
- `storage/`: Milvus client and schema management.
- `retrieval/`: Hybrid retriever and reranker.
- `generation/`: LLM interface and prompt templates.
- `evaluation/`: RAGAS metrics and synthetic data generation.
- `debugging/`: Tools for tracing and inspecting chunks.
