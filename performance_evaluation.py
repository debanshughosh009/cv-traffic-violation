#!/usr/bin/env python3
"""Evaluate event predictions against labelled JSON/JSONL ground truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def character_error_rate(expected: str, predicted: str) -> float:
    expected = expected or ""
    predicted = predicted or ""
    if not expected:
        return 0.0 if not predicted else 1.0
    previous = list(range(len(predicted) + 1))
    for row_index, expected_character in enumerate(expected, start=1):
        current = [row_index]
        for column_index, predicted_character in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (expected_character != predicted_character),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def evaluate_ocr(
    truth: Iterable[dict[str, object]],
    predictions: Iterable[dict[str, object]],
) -> dict[str, object]:
    predictions_by_id = {
        str(row["event_id"]): str(row.get("plate_text") or "")
        for row in predictions
    }
    truth_rows = list(truth)
    exact = 0
    error_rates: list[float] = []
    for row in truth_rows:
        expected = str(row.get("plate_text") or "")
        predicted = predictions_by_id.get(str(row["event_id"]), "")
        exact += int(expected == predicted)
        error_rates.append(character_error_rate(expected, predicted))
    return {
        "samples": len(truth_rows),
        "exact_match_accuracy": round(
            exact / len(truth_rows) if truth_rows else 0.0, 4
        ),
        "character_error_rate": round(
            sum(error_rates) / len(error_rates) if error_rates else 0.0, 4
        ),
    }


def _score(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def average_precision(
    ranked_predictions: list[tuple[float, bool]], positives: int
) -> float:
    if positives <= 0:
        return 0.0
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, matched) in enumerate(
        sorted(ranked_predictions, key=lambda item: item[0], reverse=True), start=1
    ):
        if matched:
            true_positives += 1
            precision_sum += true_positives / rank
    return round(precision_sum / positives, 4)


def evaluate_events(
    truth: Iterable[dict[str, object]],
    predictions: Iterable[dict[str, object]],
    frame_tolerance: int = 5,
) -> dict[str, object]:
    truth_rows = list(truth)
    prediction_rows = list(predictions)
    classes = sorted(
        {str(row["violation_type"]) for row in truth_rows + prediction_rows}
    )
    per_class: dict[str, dict[str, float | int]] = {}
    totals = [0, 0, 0]
    average_precisions: list[float] = []
    for class_name in classes:
        class_truth = [
            row for row in truth_rows if row["violation_type"] == class_name
        ]
        class_predictions = [
            row for row in prediction_rows if row["violation_type"] == class_name
        ]
        matched_truth: set[int] = set()
        tp = 0
        ranked_matches: list[tuple[float, bool]] = []
        for prediction in sorted(
            class_predictions,
            key=lambda row: float(row.get("violation_confidence", 1.0)),
            reverse=True,
        ):
            candidates = [
                (index, truth_row)
                for index, truth_row in enumerate(class_truth)
                if index not in matched_truth
                and truth_row.get("source") == prediction.get("source")
                and abs(
                    int(truth_row["frame_index"]) - int(prediction["frame_index"])
                )
                <= frame_tolerance
            ]
            if candidates:
                best_index, _ = min(
                    candidates,
                    key=lambda item: abs(
                        int(item[1]["frame_index"]) - int(prediction["frame_index"])
                    ),
                )
                matched_truth.add(best_index)
                tp += 1
                ranked_matches.append(
                    (float(prediction.get("violation_confidence", 1.0)), True)
                )
            else:
                ranked_matches.append(
                    (float(prediction.get("violation_confidence", 1.0)), False)
                )
        fp = len(class_predictions) - tp
        fn = len(class_truth) - tp
        per_class[class_name] = _score(tp, fp, fn)
        per_class[class_name]["average_precision"] = average_precision(
            ranked_matches, len(class_truth)
        )
        if class_truth:
            average_precisions.append(
                float(per_class[class_name]["average_precision"])
            )
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
    return {
        "frame_tolerance": frame_tolerance,
        "per_class": per_class,
        "micro": _score(*totals),
        "mAP": round(
            sum(average_precisions) / len(average_precisions)
            if average_precisions
            else 0.0,
            4,
        ),
    }


def load_records(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frame-tolerance", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--evaluate-ocr",
        action="store_true",
        help="Also compare plate_text values by event_id.",
    )
    args = parser.parse_args()
    result = evaluate_events(
        truth := load_records(args.ground_truth),
        predictions := load_records(args.predictions),
        args.frame_tolerance,
    )
    if args.evaluate_ocr:
        result["ocr"] = evaluate_ocr(truth, predictions)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
