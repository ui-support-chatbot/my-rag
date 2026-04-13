# Step-by-Step Guide: Running MMORE Pipeline (Process to RAG) using Docker

This document provides the exact, working Docker commands to run the entire MMORE pipeline on the server (like `riset-01`) using the `mmore-all` Docker image. It includes fixes for known directory mount errors and the `ModuleNotFoundError: No module named 'dask'` issue when using local `pyproject.toml` extras.

> **Important Setup Note**: 
> All `docker run` commands must specify `--pids-limit -1` and `--security-opt seccomp=unconfined` to prevent the container from crashing when OpenBLAS attempts to spawn math threads.

---

## 1. Stage 1: Data Processing (PDF → Markdown)

This step ingests documents from the inputs folder, extracts the text, drops unnecessary images based on your config, and outputs structured markdown to your outputs directory.

Because the `mmore-all` image can sometimes be missing the `[process]` extras (due to self-referencing issues in `-e .[all]` during pip install), we override the entrypoint to dynamically `pip install` the missing `dask` and other processor packages right before running `mmore process`.

```bash
docker run --rm \
  --entrypoint bash \
  --pids-limit -1 --security-opt seccomp=unconfined \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore/data_ingestion:/app/data_ingestion \
  mmore-all -c "pip install --no-cache-dir -e '.[process]' && python -m mmore process --config-file /app/configs/sk_rektor_process_docker.yaml"
```

*Note: Ensure your `data_path` inside `sk_rektor_process_docker.yaml` matches `/app/data_ingestion/inputs`!*

---

## 2. Stage 2: Post-Processing (Chunking)

This step splits the unified markdown file generated above into smaller sentence/token chunks suitable for embeddings.

```bash
docker run --rm \
  --entrypoint bash \
  --pids-limit -1 --security-opt seccomp=unconfined \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore/data_ingestion:/app/data_ingestion \
  mmore-all -c "pip install --no-cache-dir -e '.[process]' && python -m mmore postprocess --config-file /app/configs/sk_rektor_postprocess_docker.yaml --input-data /app/data_ingestion/outputs/process/merged/merged_results.jsonl"
```

---

## 3. Stage 3: Indexing (Embedding to Vector DB)

This step creates vector embeddings for the text chunks produced in Stage 2 and inserts them into the Milvus vector database. 
You do not need the `pip install` hack here because the `mmore-all` image natively handles `index` capabilities.

Ensure that your `milvus` configuration inside `sk_rektor_index_docker.yaml` points to the correct location (e.g., `http://localhost:19530` for standalone, or the `--network host` mapped SQLite `.db` path).

```bash
docker run --rm \
  --pids-limit -1 --security-opt seccomp=unconfined \
  --network host \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore/data_ingestion:/app/data_ingestion \
  -v ~/mmore_data:/app/mmore_data \
  mmore-all index \
  --config-file /app/configs/sk_rektor_index_docker.yaml \
  --documents-path /app/data_ingestion/outputs/postprocess/merged/results.jsonl
```

*Note: Double check that `--documents-path` points exactly to the output location created by the Post-Process step.*

---

## 4. Stage 4: Starting the RAG API

Finally, connect your Milvus database and your Reranker/LLM to run the RAG API server. This container should run in detached mode (`-d`) in the background.

```bash
# 1. Ensure any old instance is killed
docker rm -f mmore-rag

# 2. Run the persistent RAG API container
docker run -d --name mmore-rag \
  --pids-limit -1 \
  --security-opt seccomp=unconfined \
  --network host \
  -v ~/mmore/configs:/app/configs \
  -v ~/mmore_data:/app/mmore_data \
  -e OPENAI_API_KEY=dummy \
  --restart unless-stopped \
  mmore-all rag --config-file /app/configs/sk_rektor_rag_docker.yaml
```

Once running, you can monitor the API logs:
```bash
docker logs -f mmore-rag
```

You can now query the chatbot by sending a `POST` request to `http://<SERVER_IP>:8000/rag`.
