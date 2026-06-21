#!/usr/bin/env python3
"""Validate specialized model manifests before running the vision pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def canonical_model_label(label: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    mappings = {
        "driver": "driver",
        "car_driver": "driver",
        "motorbike_driver": "driver",
        "motorcycle_driver": "driver",
        "rider": "rider",
        "no_helmet": "helmet_non_compliance",
        "without_helmet": "helmet_non_compliance",
        "helmetless": "helmet_non_compliance",
        "no_seatbelt": "seatbelt_non_compliance",
        "without_seatbelt": "seatbelt_non_compliance",
        "seatbelt_missing": "seatbelt_non_compliance",
        "license_plate": "license_plate",
        "number_plate": "license_plate",
        "plate": "license_plate",
    }
    return mappings.get(normalized)


def validate_model_spec(spec: dict[str, object], base_dir: Path) -> dict[str, object]:
    issues: list[str] = []
    weights_value = str(spec.get("weights", "")).strip()
    weights = (base_dir / weights_value).resolve() if weights_value else None
    if weights is None or not weights.is_file():
        issues.append("weights_not_found")
    required = [str(label) for label in spec.get("required_labels", [])]
    unknown = [label for label in required if canonical_model_label(label) is None]
    if unknown:
        issues.append("unknown_required_labels")
    checksum = None
    if weights and weights.is_file():
        digest = hashlib.sha256()
        with weights.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        expected = spec.get("sha256")
        if expected and str(expected).lower() != checksum:
            issues.append("checksum_mismatch")
    return {
        "task": spec.get("task"),
        "weights": str(weights) if weights else None,
        "sha256": checksum,
        "required_labels": required,
        "ready": not issues,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    specs = raw.get("models", [])
    results = [
        validate_model_spec(spec, args.manifest.parent) for spec in specs
    ]
    output = {"ready": all(result["ready"] for result in results), "models": results}
    print(json.dumps(output, indent=2))
    if not output["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
