"""Build or inspect the scoped instruction index without changing source files."""

import argparse
import json

from app.core.config import get_settings
from app.db import session_factory
from app.services.rag.catalog_worker import synchronize_knowledge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-read all originals to verify changed content")
    args = parser.parse_args(argv)
    try:
        result = synchronize_knowledge(session_factory(), get_settings(), refresh=args.refresh)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result.get("complete") else 1
    except Exception as error:
        print(json.dumps({"status": "error", "error_code": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
