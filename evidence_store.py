"""SQLite-backed searchable violation evidence storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                violation_type TEXT NOT NULL,
                rule_id TEXT,
                source TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_ms REAL NOT NULL,
                track_id INTEGER,
                vehicle_class TEXT,
                plate_text TEXT,
                detection_confidence REAL,
                violation_confidence REAL NOT NULL,
                evidence_image TEXT,
                details_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending',
                reviewer TEXT,
                review_notes TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(events)").fetchall()
        }
        migrations = {
            "review_status": "TEXT NOT NULL DEFAULT 'pending'",
            "reviewer": "TEXT",
            "review_notes": "TEXT",
            "reviewed_at": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in existing_columns:
                self.connection.execute(
                    f"ALTER TABLE events ADD COLUMN {column} {definition}"
                )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_time "
            "ON events(violation_type, timestamp_ms)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_text)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_source_frame "
            "ON events(source, frame_index)"
        )
        self.connection.commit()

    def insert_event(self, event: dict[str, object]) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO events (
                event_id, violation_type, rule_id, source, frame_index,
                timestamp_ms, track_id, vehicle_class, plate_text,
                detection_confidence, violation_confidence, evidence_image,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["violation_type"],
                event.get("rule_id"),
                event["source"],
                event["frame_index"],
                event["timestamp_ms"],
                event.get("track_id"),
                event.get("vehicle_class"),
                event.get("plate_text"),
                event.get("detection_confidence"),
                event["violation_confidence"],
                event.get("evidence_image"),
                json.dumps(event.get("details", {}), separators=(",", ":")),
            ),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def search(
        self,
        violation_type: str | None = None,
        plate_text: str | None = None,
        source: str | None = None,
        review_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        values: list[object] = []
        if violation_type:
            clauses.append("violation_type = ?")
            values.append(violation_type)
        if plate_text:
            clauses.append("plate_text LIKE ?")
            values.append(f"%{plate_text.upper()}%")
        if source:
            clauses.append("source = ?")
            values.append(source)
        if review_status:
            clauses.append("review_status = ?")
            values.append(review_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, limit))
        rows = self.connection.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp_ms DESC LIMIT ?",
            values,
        ).fetchall()
        return [self._row(row) for row in rows]

    def update_review(
        self,
        event_id: str,
        status: str,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> bool:
        allowed = {"pending", "confirmed", "rejected", "needs_review"}
        if status not in allowed:
            raise ValueError(
                f"Invalid review status {status!r}; expected one of {sorted(allowed)}."
            )
        cursor = self.connection.execute(
            """
            UPDATE events
            SET review_status = ?, reviewer = ?, review_notes = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE event_id = ?
            """,
            (status, reviewer, notes, event_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def all_events(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM events ORDER BY timestamp_ms"
        ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, object]:
        record = dict(row)
        record["details"] = json.loads(str(record.pop("details_json")))
        return record

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
