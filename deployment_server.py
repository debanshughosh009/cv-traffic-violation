#!/usr/bin/env python3
"""Small authenticated HTTP API for health, search, and human review."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analytics_report import build_summary
from evidence_store import EvidenceStore
from security_controls import AuditLogger, token_matches


class TrafficAPIHandler(BaseHTTPRequestHandler):
    database: Path
    token_hash: str
    audit_log: Path

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        return token_matches(header[7:], self.token_hash)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        query = parse_qs(parsed.query)
        with EvidenceStore(self.database) as store:
            if parsed.path == "/events":
                rows = store.search(
                    violation_type=query.get("type", [None])[0],
                    plate_text=query.get("plate", [None])[0],
                    source=query.get("source", [None])[0],
                    review_status=query.get("review_status", [None])[0],
                    limit=int(query.get("limit", ["100"])[0]),
                )
                self._json(200, rows)
                return
            if parsed.path == "/summary":
                self._json(200, build_summary(store.all_events()))
                return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 3 or parts[0] != "events" or parts[2] != "review":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            with EvidenceStore(self.database) as store:
                changed = store.update_review(
                    parts[1],
                    str(payload["status"]),
                    str(payload.get("reviewer", "api")),
                    payload.get("notes"),
                )
            if not changed:
                self._json(404, {"error": "event_not_found"})
                return
            AuditLogger(self.audit_log).write(
                "review.updated",
                actor=str(payload.get("reviewer", "api")),
                target=parts[1],
                details={"status": payload["status"]},
            )
            self._json(200, {"updated": True})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audit-log", type=Path, default=Path("runs/audit.jsonl"))
    args = parser.parse_args()
    token_hash = os.environ.get("TRAFFIC_API_TOKEN_HASH")
    if not token_hash:
        raise SystemExit("Set TRAFFIC_API_TOKEN_HASH before starting the server.")
    handler = type(
        "ConfiguredTrafficAPIHandler",
        (TrafficAPIHandler,),
        {
            "database": args.database,
            "token_hash": token_hash,
            "audit_log": args.audit_log,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Traffic API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
