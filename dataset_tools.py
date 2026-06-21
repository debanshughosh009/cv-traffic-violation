#!/usr/bin/env python3
"""Create and validate labelled YOLO datasets for specialized traffic models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    class_names: tuple[str, ...]
    splits: tuple[str, ...] = ("train", "val", "test")


def write_yolo_dataset_template(root: Path, class_names: list[str]) -> Path:
    if not class_names:
        raise ValueError("At least one class name is required.")
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    yaml_path = root / "dataset.yaml"
    names = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(class_names)
    )
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    (root / "labels.json").write_text(
        json.dumps({"classes": class_names}, indent=2), encoding="utf-8"
    )
    return yaml_path


def validate_dataset(manifest: DatasetManifest) -> dict[str, object]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count = 0
    label_count = 0
    missing_labels = 0
    invalid_labels = 0
    empty_splits: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split in manifest.splits:
        image_dir = manifest.root / "images" / split
        label_dir = manifest.root / "labels" / split
        images = (
            [
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in image_extensions
            ]
            if image_dir.is_dir()
            else []
        )
        if not images:
            empty_splits.append(split)
        split_label_count = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                missing_labels += 1
                continue
            split_label_count += 1
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                except (ValueError, IndexError):
                    invalid_labels += 1
                    continue
                if (
                    len(coordinates) != 4
                    or not 0 <= class_id < len(manifest.class_names)
                    or not all(0.0 <= value <= 1.0 for value in coordinates)
                ):
                    invalid_labels += 1
        image_count += len(images)
        label_count += split_label_count
        split_counts[split] = {
            "images": len(images),
            "labels": split_label_count,
        }
    return {
        "ready": not missing_labels and not invalid_labels and not empty_splits,
        "images": image_count,
        "labels": label_count,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "empty_splits": empty_splits,
        "splits": split_counts,
        "classes": list(manifest.class_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--classes", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--classes", required=True)
    args = parser.parse_args()
    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    if args.command == "create":
        print(write_yolo_dataset_template(args.root, classes))
    else:
        result = validate_dataset(
            DatasetManifest(args.root, tuple(classes))
        )
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
