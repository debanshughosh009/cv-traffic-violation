#!/usr/bin/env python3
"""Reproducible Ultralytics training/validation/export orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_training_command(
    data: Path,
    base_model: str,
    epochs: int,
    image_size: int,
    project: Path,
    name: str,
    seed: int,
    device: str | None = None,
) -> list[str]:
    command = [
        "yolo",
        "detect",
        "train",
        f"data={data}",
        f"model={base_model}",
        f"epochs={epochs}",
        f"imgsz={image_size}",
        f"project={project}",
        f"name={name}",
        f"seed={seed}",
        "deterministic=True",
    ]
    if device:
        command.append(f"device={device}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("runs/training"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export", choices=("onnx", "engine", "openvino"))
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")
    command = build_training_command(
        args.data,
        args.base_model,
        args.epochs,
        args.imgsz,
        args.project,
        args.name,
        args.seed,
        args.device,
    )
    run_manifest = {
        "command": command,
        "dataset": str(args.data.resolve()),
        "base_model": args.base_model,
        "seed": args.seed,
    }
    args.project.mkdir(parents=True, exist_ok=True)
    manifest_path = args.project / f"{args.name}-run.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    best = args.project / args.name / "weights" / "best.pt"
    subprocess.run(
        ["yolo", "detect", "val", f"model={best}", f"data={args.data}"],
        check=True,
    )
    if args.export:
        subprocess.run(
            ["yolo", "export", f"model={best}", f"format={args.export}"],
            check=True,
        )


if __name__ == "__main__":
    main()
