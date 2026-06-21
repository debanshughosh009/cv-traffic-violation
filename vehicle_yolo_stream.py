#!/usr/bin/env python3
"""Run YOLO vehicle detection on a webcam, video file, or network stream."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect vehicles in a live video stream using a pretrained YOLO model."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index like 0, a video path, or an RTSP/HTTP stream URL.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights to use. Ultralytics downloads yolov8n.pt on first run if needed.",
    )
    parser.add_argument(
        "--vehicle-classes",
        default=",".join(DEFAULT_VEHICLE_CLASSES),
        help="Comma-separated COCO class names to keep, for example car,motorcycle,bus,truck.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as cpu, cuda, cuda:0, or mps. Defaults to Ultralytics auto-select.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for an annotated output video, for example runs/vehicles.mp4.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the preview window. Useful on servers or when only saving output.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: tqdm. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return cv2, YOLO, tqdm


def normalize_source(source: str) -> int | str:
    """Convert webcam-like numeric strings to integers for cv2.VideoCapture."""
    if source.isdigit() and not Path(source).exists():
        return int(source)
    return source


def parse_class_names(raw: str) -> tuple[str, ...]:
    names = tuple(name.strip().lower() for name in raw.split(",") if name.strip())
    if not names:
        raise SystemExit("At least one vehicle class is required.")
    return names


def resolve_class_ids(model_names: dict[int, str] | list[str], requested: Iterable[str]) -> list[int]:
    names_by_id = dict(model_names) if isinstance(model_names, dict) else dict(enumerate(model_names))
    ids_by_name = {name.lower(): class_id for class_id, name in names_by_id.items()}

    missing = sorted(set(requested) - set(ids_by_name))
    if missing:
        available = ", ".join(names_by_id[index] for index in sorted(names_by_id))
        raise SystemExit(
            f"Unknown class name(s): {', '.join(missing)}.\n"
            f"Available model classes: {available}"
        )

    return [ids_by_name[name] for name in requested]


def should_show_window(no_show: bool) -> bool:
    if no_show:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("No DISPLAY found; running without a preview window.", file=sys.stderr)
        return False
    return True


def create_video_writer(cv2, output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps or 30.0, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video for writing: {output_path}")
    return writer


def draw_detections(cv2, frame, boxes, class_names: dict[int, str]) -> int:
    vehicle_count = 0

    for box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = f"{class_names[class_id]} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + baseline + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - baseline - 4),
            (x1 + label_size[0] + 8, label_y + baseline - 2),
            (0, 180, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        vehicle_count += 1

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )
    return vehicle_count


def run() -> None:
    args = parse_args()
    cv2, YOLO, tqdm = import_dependencies()

    model = YOLO(args.model)
    requested_classes = parse_class_names(args.vehicle_classes)
    class_ids = resolve_class_ids(model.names, requested_classes)
    class_names = dict(model.names)

    source = normalize_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = None
    show_window = should_show_window(args.no_show)
    frame_count = 0
    start_time = time.perf_counter()
    raw_total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = (
        int(raw_total_frames)
        if raw_total_frames > 0 and raw_total_frames < float("inf")
        else None
    )
    progress = tqdm(
        total=total_frames,
        desc="Detecting vehicles",
        unit="frame",
        dynamic_ncols=True,
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if args.output and writer is None:
                height, width = frame.shape[:2]
                writer = create_video_writer(cv2, args.output, fps, width, height)

            result = model.predict(
                frame,
                classes=class_ids,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            count = draw_detections(cv2, frame, result.boxes, class_names)
            frame_count += 1
            progress.update(1)
            progress.set_postfix(vehicles=count, refresh=False)

            elapsed = max(time.perf_counter() - start_time, 1e-9)
            cv2.putText(
                frame,
                f"FPS: {frame_count / elapsed:.1f}",
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 0),
                2,
                cv2.LINE_AA,
            )

            if writer:
                writer.write(frame)

            if show_window:
                cv2.imshow("YOLO Vehicle Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        progress.close()
        capture.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"Finished. Processed {frame_count} frame(s).")
    if args.output:
        print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    run()
