#!/usr/bin/env python3
"""
CLI for ingesting documents into the RAG system.

Usage examples:
  python ingest.py --file /path/to/doc.pdf
  python ingest.py --dir ./data/raw

The script prints a JSON result and exits with non-zero on error.
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import get_logger
from app.services.ingestion import get_ingestion_service

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Path to a single file to ingest")
    group.add_argument("--dir", "-d", help="Directory containing documents to ingest")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size override")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Optional chunk overlap override")

    args = parser.parse_args()

    service = get_ingestion_service()

    try:
        if args.file:
            result = service.ingest_file(args.file, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        else:
            result = service.ingest_directory(args.dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

        print(json.dumps(result, indent=2, default=str))

        if isinstance(result, dict) and result.get("status") == "error":
            logger.error("Ingestion failed: %s", result.get("message"))
            return 1
        return 0

    except Exception as e:
        logger.exception("Unhandled error during ingestion: %s", e)
        print(json.dumps({"status": "error", "message": str(e)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
