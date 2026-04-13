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


if __name__ == "__main__":
    main()
