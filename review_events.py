#!/usr/bin/env python3
"""List and update human-review decisions for violation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_store import EvidenceStore
from security_controls import AuditLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, default=Path("runs/audit.jsonl"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--status", default="pending")
    listing.add_argument("--limit", type=int, default=50)
    update = subparsers.add_parser("update")
    update.add_argument("--event-id", required=True)
    update.add_argument(
        "--status",
        required=True,
        choices=("pending", "confirmed", "rejected", "needs_review"),
    )
    update.add_argument("--reviewer", required=True)
    update.add_argument("--notes")
    args = parser.parse_args()
    audit = AuditLogger(args.audit_log)
    with EvidenceStore(args.database) as store:
        if args.command == "list":
            print(
                json.dumps(
                    store.search(review_status=args.status, limit=args.limit), indent=2
                )
            )
            return
        changed = store.update_review(
            args.event_id, args.status, args.reviewer, args.notes
        )
    if not changed:
        raise SystemExit(f"Event not found: {args.event_id}")
    audit.write(
        "review.updated",
        actor=args.reviewer,
        target=args.event_id,
        details={"status": args.status, "notes": args.notes},
    )
    print(f"{args.event_id}: {args.status}")
