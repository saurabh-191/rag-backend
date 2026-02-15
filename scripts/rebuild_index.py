#!/usr/bin/env python3
"""
CLI to rebuild the vector index from documents.

Usage:
  python rebuild_index.py --dir ./data/raw
If no directory is provided, the default path from configuration is used.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import get_logger
from app.services.ingestion import get_ingestion_service
from app.core.config import settings

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild vector index from documents")
    parser.add_argument("--dir", "-d", help="Directory containing documents (optional)")
    args = parser.parse_args()

    directory: Optional[str] = args.dir or settings.DATA_RAW_PATH
    service = get_ingestion_service()

    try:
        result = service.rebuild_index(directory)
        print(json.dumps(result, indent=2, default=str))
        if isinstance(result, dict) and result.get("status") == "error":
            logger.error("Rebuild failed: %s", result.get("message"))
            return 1
        return 0

    except Exception as e:
        logger.exception("Unhandled error during rebuild: %s", e)
        print(json.dumps({"status": "error", "message": str(e)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
