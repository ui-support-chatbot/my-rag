import argparse
import json
import sys
from pathlib import Path

from pipeline import RAGPipeline
from config import RAGConfig


def main():
    parser = argparse.ArgumentParser(
        description="MyRAG CLI for debugging and inspection"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--config", required=True, help="Path to config YAML")
    query_parser.add_argument("--query", required=True, help="Query string")
    query_parser.add_argument("--doc-ids", nargs="*", help="Filter by document IDs")
    query_parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )

    find_parser = subparsers.add_parser(
        "find-keyword", help="Find chunks containing keyword"
    )
    find_parser.add_argument("--config", required=True, help="Path to config YAML")
    find_parser.add_argument("--keyword", required=True, help="Keyword to search")
    find_parser.add_argument("--doc-id", help="Filter by document ID")
    find_parser.add_argument(
        "--case-sensitive", action="store_true", help="Case sensitive search"
    )

    trace_parser = subparsers.add_parser("trace", help="Trace query with keyword check")
    trace_parser.add_argument("--config", required=True, help="Path to config YAML")
    trace_parser.add_argument("--query", required=True, help="Query string")
    trace_parser.add_argument(
        "--check-keyword", required=True, help="Keyword to verify in results"
    )
    trace_parser.add_argument("--doc-ids", nargs="*", help="Filter by document IDs")
    trace_parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate RAG with synthetic QA")
    eval_parser.add_argument("--config", required=True, help="Path to config YAML")
    eval_parser.add_argument("--questions", nargs="*", help="Questions to evaluate")
    eval_parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic QA"
    )
    eval_parser.add_argument("--output", help="Output file for results")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--config", required=True, help="Path to config YAML")
    ingest_parser.add_argument("--paths", nargs="+", help="File paths to ingest")
    ingest_parser.add_argument("--directory", help="Directory to ingest")
    ingest_parser.add_argument("--prefix", default="doc", help="Document ID prefix")

    inspect_chunks_parser = subparsers.add_parser("inspect-chunks", help="Inspect chunks before embedding")
    inspect_chunks_parser.add_argument("--config", required=True, help="Path to config YAML")
    inspect_chunks_parser.add_argument("--directory", required=True, help="Directory to process")
    inspect_chunks_parser.add_argument("--output-file", help="Save chunks to file")
    inspect_chunks_parser.add_argument("--show-stats", action="store_true", help="Show chunking statistics")
    inspect_chunks_parser.add_argument("--filter-keyword", help="Filter chunks containing keyword")

    debug_query_parser = subparsers.add_parser("debug-query", help="Debug query with full pipeline inspection")
    debug_query_parser.add_argument("--config", required=True, help="Path to config YAML")
    debug_query_parser.add_argument("--query", required=True, help="Query string")
    debug_query_parser.add_argument("--show-stages", action="store_true", help="Show results at each stage")
    debug_query_parser.add_argument("--output-format", choices=["json", "text", "detailed"], default="json", help="Output format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config = RAGConfig.from_yaml(args.config)
    rag = RAGPipeline.from_config(config)

    if args.command == "query":
        result = rag.query(
            query=args.query,
            doc_ids=args.doc_ids,
            k=args.k,
        )
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "context": result.context,
                    "num_docs": result.metadata["num_docs"],
                },
                indent=2,
            )
        )

    elif args.command == "find-keyword":
        matches = rag.find_keyword(
            keyword=args.keyword,
            doc_id=args.doc_id,
        )
        print(
            json.dumps(
                {
                    "keyword": args.keyword,
                    "num_matches": len(matches),
                    "matches": matches,
                },
                indent=2,
            )
        )

    elif args.command == "trace":
        result = rag.query_with_keyword_check(
            query=args.query,
            check_keyword=args.check_keyword,
            doc_ids=args.doc_ids,
            k=args.k,
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "eval":
        questions = args.questions
        if args.synthetic:
            qa_pairs = rag.generate_synthetic_qa(
                paths=args.paths if hasattr(args, "paths") else None,
            )
            questions = [p["question"] for p in qa_pairs]
            print(f"Generated {len(questions)} synthetic questions")

        results = rag.evaluate(questions=questions)
        output = args.output or "eval_results.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output}")

    elif args.command == "ingest":
        count = rag.ingest(
            paths=args.paths,
            directory=args.directory,
            doc_id_prefix=args.prefix,
        )
        print(f"Indexed {count} chunks")

    elif args.command == "inspect-chunks":
        chunks = rag.save_chunks_before_embedding(
            directory=args.directory,
            output_file=args.output_file
        )
        
        if args.filter_keyword:
            filtered_chunks = [c for c in chunks if args.filter_keyword.lower() in c.text.lower()]
            print(f"Found {len(filtered_chunks)} chunks containing '{args.filter_keyword}':")
            for i, chunk in enumerate(filtered_chunks):
                print(f"\n--- Chunk {i+1} ---")
                print(f"Doc ID: {chunk.doc_id}")
                print(f"Breadcrumb: {chunk.breadcrumb}")
                print(f"Page: {chunk.page_number}")
                print(f"Filename: {chunk.filename}")
                print(f"Text Preview: {chunk.text[:200]}...")
        else:
            print(f"Total chunks: {len(chunks)}")
            if args.show_stats:
                total_chars = sum(len(c.text) for c in chunks)
                avg_chars = total_chars / len(chunks) if chunks else 0
                print(f"Total characters: {total_chars}")
                print(f"Average characters per chunk: {avg_chars:.2f}")
                print(f"Unique documents: {len(set(c.doc_id for c in chunks))}")
                
                if chunks:
                    print("\nFirst few chunks:")
                    for i, chunk in enumerate(chunks[:3]):
                        print(f"\n--- Chunk {i+1} ---")
                        print(f"Doc ID: {chunk.doc_id}")
                        print(f"Breadcrumb: {chunk.breadcrumb}")
                        print(f"Page: {chunk.page_number}")
                        print(f"Filename: {chunk.filename}")
                        print(f"Text Preview: {chunk.text[:200]}...")

    elif args.command == "debug-query":
        print(f"Processing query: {args.query}")
        
        if args.show_stages:
            print("\n1. Retrieving documents...")
            docs = rag.retriever.retrieve(
                query=args.query,
                collection_name=rag.config.storage.collection_name,
                k=rag.config.retrieval.k,
            )
            print(f"Retrieved {len(docs)} documents after RRF fusion")
            
            print("\n2. Applying reranking...")
            reranked_docs = rag.retriever._rerank(args.query, docs)
            # Slice to rerank_top_k to match the pipeline behavior
            rerank_top_k = rag.config.retrieval.rerank_top_k
            reranked_docs = reranked_docs[:rerank_top_k]
            print(f"Reranked to top {len(reranked_docs)} documents")
            
            if args.output_format == "detailed":
                print("\nDetailed results:")
                for i, doc in enumerate(reranked_docs):
                    print(f"\n--- Document {i+1} (Score: {doc.score:.4f}) ---")
                    print(f"Doc ID: {doc.doc_id}")
                    print(f"Breadcrumb: {doc.metadata.get('breadcrumb', 'N/A')}")
                    print(f"Page: {doc.metadata.get('page_number', 'N/A')}")
                    print(f"Text Preview: {doc.text[:300]}...")
            elif args.output_format == "text":
                print("\nContext that would be sent to LLM:")
                for i, doc in enumerate(reranked_docs):
                    print(f"\n[Source {doc.metadata.get('breadcrumb', f'Doc_{i+1}')}]")
                    print(doc.text)
            else:  # json format
                result = {
                    "query": args.query,
                    "retrieved_count": len(docs),
                    "reranked_count": len(reranked_docs),
                    "documents": [
                        {
                            "doc_id": doc.doc_id,
                            "breadcrumb": doc.metadata.get("breadcrumb", "N/A"),
                            "page_number": doc.metadata.get("page_number", "N/A"),
                            "score": doc.score,
                            "text_preview": doc.text[:300] + "..." if len(doc.text) > 300 else doc.text
                        }
                        for doc in reranked_docs
                    ]
                }
                print(json.dumps(result, indent=2, default=str))
        else:
            # Just run the normal query
            result = rag.query(query=args.query)
            print(json.dumps({
                "answer": result.answer,
                "context": result.context,
                "num_docs": result.metadata["num_docs"],
            }, indent=2))


if __name__ == "__main__":
    main()
