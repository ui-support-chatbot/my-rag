# Handover Prompt: GPU Memory Optimization for RAG Retrieval Pipeline

## Problem
The current retrieval pipeline loads three models (dense, sparse, reranker) simultaneously on the same GPU, consuming approximately 8GB of VRAM. This leaves insufficient memory for a local LLM (e.g., Llama 3 8B requires ~16GB in FP16, but we can use quantization to reduce it to ~8GB, so we need to free up as much as possible).

## Goal
Reduce the GPU memory footprint of the retrieval pipeline to under 4GB, allowing the local LLM to run on the same GPU.

## Proposed Solutions
1. **Lazy Loading and Model Offloading**:
   - Load the dense model only when needed for query encoding, then move it to CPU.
   - Load the reranker model only when needed for reranking, then move it to CPU.
   - The sparse model's neural component is only required for document encoding (during ingestion). For query encoding, only the tokenizer and IDF table are needed (which are CPU-resident). Therefore, we can keep the sparse model on CPU at all times, or load it only during ingestion and then unload.

2. **8-bit Quantization**:
   - Use 8-bit quantization for the dense and reranker models to reduce their memory footprint by approximately 50%.
   - This can be done by setting `load_in_8bit=True` when loading the models with Hugging Face's `transformers` or `sentence_transformers`.

3. **Pre-compute Embeddings**:
   - Since document embeddings are static, we can pre-compute and store the dense and sparse vectors for all documents during ingestion.
   - At query time, we only need to compute the query's dense and sparse vectors (which are cheap) and then perform vector search in Milvus.
   - This eliminates the need to keep the document encoding models in memory during retrieval.

## Implementation Plan
Step 1: Modify the sparse model to run on CPU only (since its neural part is not needed for query encoding).
Step 2: Implement lazy loading for the dense and reranker models in the retriever, with methods to load and unload.
Step 3: Add 8-bit quantization options in the model loading configuration.
Step 4: (Optional) Implement pre-compute embeddings for documents if not already done.

## Expected Outcome
After implementation, the retrieval pipeline should use approximately:
   - Dense model (0.6B, 8-bit): ~0.6GB
   - Sparse model (on CPU): ~0GB GPU
   - Reranker (0.6B, 8-bit): ~0.6GB
   - Total: ~1.2GB (plus overhead for activations and Milvus, aiming for under 2GB)

This leaves ample room for a quantized LLM (e.g., Llama 3 8B in 4-bit: ~4GB) on the same GPU.

## Files to Modify
- `embedding/dense.py`: Add lazy loading and quantization.
- `embedding/sparse.py`: Change to load on CPU by default, and only load the neural part during ingestion if needed.
- `retrieval/retriever.py`: Implement lazy loading for dense and reranker models, and unload after use.
- `config_rag.yaml`: Add configuration for quantization and device mapping.

## Testing
- Verify that the retrieval results remain accurate after changes.
- Measure GPU memory usage before and after each optimization.
- Ensure that the local LLM can run without OOM errors.

## Notes
- We must be cautious about the trade-off between memory savings and latency (loading models takes time). We may want to keep models loaded in a warm state if queries are frequent.
- The sparse model's neural component is only used during ingestion, so we can load it on GPU only during ingestion and then unload.