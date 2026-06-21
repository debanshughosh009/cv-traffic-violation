#!/usr/bin/env python3
"""Search the local violation evidence database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_store import EvidenceStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--type", dest="violation_type")
    parser.add_argument("--plate")
    parser.add_argument("--source")
    parser.add_argument("--review-status")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    with EvidenceStore(args.database) as store:
        rows = store.search(
            violation_type=args.violation_type,
            plate_text=args.plate,
            source=args.source,
            review_status=args.review_status,
            limit=args.limit,
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
