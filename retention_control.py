#!/usr/bin/env python3
"""Apply evidence-retention policy to database records and local images."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evidence_store import EvidenceStore
from security_controls import AuditLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--actor", default="retention-job")
    parser.add_argument("--audit-log", type=Path, default=Path("runs/audit.jsonl"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be at least 1.")
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    with EvidenceStore(args.database) as store:
        rows = store.connection.execute(
            "SELECT event_id, evidence_image, created_at FROM events "
            "WHERE datetime(created_at) < datetime(?)",
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchall()
        print(f"Matched {len(rows)} event(s) older than {args.days} day(s).")
        if not args.apply:
            print("Dry run only. Add --apply to delete.")
            return
        for row in rows:
            relative = row["evidence_image"]
            if relative:
                candidate = (args.evidence_root / str(relative)).resolve()
                root = args.evidence_root.resolve()
                if candidate == root or root not in candidate.parents:
                    raise SystemExit(f"Unsafe evidence path: {candidate}")
                if candidate.is_file():
                    candidate.unlink()
        store.connection.execute(
            "DELETE FROM events WHERE datetime(created_at) < datetime(?)",
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        store.connection.commit()
    AuditLogger(args.audit_log).write(
        "retention.applied",
        actor=args.actor,
        details={"days": args.days, "deleted_events": len(rows)},
    )
