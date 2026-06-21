#!/usr/bin/env python3
"""Generate JSON, CSV, and self-contained HTML reports from the event database."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

from evidence_store import EvidenceStore


def build_time_series(
    events: Iterable[dict[str, object]], bucket_seconds: int = 300
) -> list[dict[str, object]]:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be greater than zero.")
    bucket_ms = bucket_seconds * 1000
    buckets: dict[int, Counter[str]] = {}
    for event in events:
        bucket = int(float(event["timestamp_ms"]) // bucket_ms) * bucket_ms
        buckets.setdefault(bucket, Counter())[str(event["violation_type"])] += 1
    return [
        {
            "bucket_start_ms": bucket,
            "bucket_end_ms": bucket + bucket_ms,
            "total": sum(counts.values()),
            "by_violation": dict(sorted(counts.items())),
        }
        for bucket, counts in sorted(buckets.items())
    ]


def build_summary(events: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(events)
    by_violation = Counter(str(row["violation_type"]) for row in rows)
    by_source = Counter(str(row.get("source") or "unknown") for row in rows)
    by_review_status = Counter(
        str(row.get("review_status") or "pending") for row in rows
    )
    plates = Counter(
        str(row["plate_text"]) for row in rows if row.get("plate_text")
    )
    confidences = [float(row["violation_confidence"]) for row in rows]
    return {
        "total_events": len(rows),
        "by_violation": dict(sorted(by_violation.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_review_status": dict(sorted(by_review_status.items())),
        "top_plates": [[plate, count] for plate, count in plates.most_common(10)],
        "average_confidence": round(mean(confidences), 4) if confidences else 0.0,
        "first_timestamp_ms": min(
            (float(row["timestamp_ms"]) for row in rows), default=None
        ),
        "last_timestamp_ms": max(
            (float(row["timestamp_ms"]) for row in rows), default=None
        ),
        "time_series": build_time_series(rows),
    }


def write_reports(
    events: list[dict[str, object]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(events)
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_dir / "events.csv"
    fields = [
        "event_id",
        "violation_type",
        "rule_id",
        "source",
        "frame_index",
        "timestamp_ms",
        "track_id",
        "vehicle_class",
        "plate_text",
        "violation_confidence",
        "evidence_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)

    cards = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in summary["by_violation"].items()
    )
    source_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in summary["by_source"].items()
    )
    trend_rows = "".join(
        "<tr>"
        f"<td>{bucket['bucket_start_ms'] / 1000:.0f}s</td>"
        f"<td>{bucket['total']}</td>"
        f"<td>{html.escape(json.dumps(bucket['by_violation']))}</td>"
        "</tr>"
        for bucket in summary["time_series"]
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(event.get('violation_type', '')))}</td>"
        f"<td>{html.escape(str(event.get('plate_text') or ''))}</td>"
        f"<td>{float(event.get('timestamp_ms', 0)) / 1000:.2f}s</td>"
        f"<td>{float(event.get('violation_confidence', 0)):.2f}</td>"
        "</tr>"
        for event in events[-200:]
    )
    html_path = output_dir / "report.html"
    html_path.write_text(
        f"""<!doctype html><meta charset="utf-8">
<title>Traffic Violation Report</title>
<style>
body{{font:16px system-ui;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.55rem;text-align:left}}
th{{background:#172033;color:white}}.metric{{font-size:2rem;font-weight:700}}
</style>
<h1>Traffic Violation Report</h1>
<p class="metric">{summary['total_events']} events</p>
<p>Average confidence: {summary['average_confidence']}</p>
<h2>Violations by type</h2><table><tr><th>Type</th><th>Count</th></tr>{cards}</table>
<h2>Events by source/camera</h2>
<table><tr><th>Source</th><th>Count</th></tr>{source_rows}</table>
<h2>Time trends</h2>
<table><tr><th>Bucket start</th><th>Total</th><th>Breakdown</th></tr>{trend_rows}</table>
<h2>Recent events</h2>
<table><tr><th>Violation</th><th>Plate</th><th>Time</th><th>Confidence</th></tr>{rows}</table>
""",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "html": html_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reports"))
    args = parser.parse_args()
    with EvidenceStore(args.database) as store:
        paths = write_reports(store.all_events(), args.output_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
