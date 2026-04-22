# MyRAG - Modular RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed for research and extensibility.

## Features

- **Advanced Ingestion**: PDF parsing via Docling with hierarchical chunking, plus HTML extraction via Trafilatura with standard overlapping text chunks.
- **Hybrid Search**: Combines Dense (BGE) and Sparse (SPLADE) embeddings using Milvus.
- **Reranking**: Integration with a dedicated GGUF reranker service to improve retrieval precision.
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

### 3. Safe index rebuild
When parser or chunker behavior changes, build a shadow collection with a fresh ingestion state instead of wiping the live collection:

```bash
python src/my_rag/cli.py rebuild-index --config config_rag.yaml --directory ./data/docs
```

After validation, print the promotion patch:

```bash
python src/my_rag/cli.py promote-index \
  --collection-name documents_rebuild_YYYYMMDD_HHMMSS \
  --state-path storage/ingestion_state_rebuild_YYYYMMDD_HHMMSS.json
```

### 4. Querying
```bash
python src/my_rag/cli.py query --config config_rag.yaml --query "What is the main topic?"
```

### 5. Debugging
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

## Known Limitations

- **Chunker Config**: The `chunk_size` and `chunk_overlap` settings in the YAML files are currently ignored by the `HierarchicalChunker` implementation in favor of structural boundaries. This is planned for a future update.
- **VRAM Competition**: Running long-context reranking (8k tokens) on consumer GPUs alongside a large LLM may require careful GPU allocation.
