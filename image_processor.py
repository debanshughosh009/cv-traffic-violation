#!/usr/bin/env python3
"""Process individual traffic photographs or image directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advanced_preprocessing import enhance_adverse_frame
from comprehensive_traffic_system import (
    TARGET_CLASSES,
    attribute_events,
    manifest_models,
    plate_texts,
    role_detections,
    tracked_objects,
    triple_riding_events,
)
from evidence_store import EvidenceStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_images(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for source in inputs:
        if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            paths.add(source)
        elif source.is_dir():
            paths.update(
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(paths, key=lambda path: str(path).lower())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--helmet-model")
    parser.add_argument("--seatbelt-model")
    parser.add_argument("--driver-model")
    parser.add_argument("--plate-model")
    parser.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/images"))
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies from requirements.txt.") from exc
    images = discover_images(args.inputs)
    if not images:
        raise SystemExit("No supported images found.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = manifest_models(args.model_manifest)
    base_model = YOLO(args.model)
    models = {
        "helmet": YOLO(args.helmet_model or manifest["helmet"])
        if args.helmet_model or manifest.get("helmet")
        else None,
        "seatbelt": YOLO(args.seatbelt_model or manifest["seatbelt"])
        if args.seatbelt_model or manifest.get("seatbelt")
        else None,
        "driver": YOLO(args.driver_model or manifest["driver"])
        if args.driver_model or manifest.get("driver")
        else None,
        "plate": YOLO(args.plate_model or manifest["license_plate"])
        if args.plate_model or manifest.get("license_plate")
        else None,
    }
    names = dict(base_model.names)
    class_ids = [
        class_id
        for class_id, name in names.items()
        if name.lower() in TARGET_CLASSES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    store = EvidenceStore(args.output_dir / "events.db")
    event_count = 0
    try:
        with events_path.open("w", encoding="utf-8") as events_file:
            for image_index, path in enumerate(images):
                original = cv2.imread(str(path))
                if original is None:
                    continue
                frame, quality = enhance_adverse_frame(
                    cv2, original, config.get("preprocessing", {})
                )
                height, width = frame.shape[:2]
                result = base_model.predict(
                    frame,
                    classes=class_ids,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )[0]
                objects = tracked_objects(result, names, width, height)
                vehicles = [item for item in objects if item.class_name != "person"]
                events = triple_riding_events(objects, width, height, 0.0)
                events += attribute_events(
                    models["helmet"], frame, 0.0, vehicles, args.device
                )
                events += attribute_events(
                    models["seatbelt"], frame, 0.0, vehicles, args.device
                )
                roles = role_detections(models["driver"], frame, args.device)
                plates = plate_texts(
                    models["plate"], frame, vehicles, args.ocr, args.device
                )
                for item in objects:
                    x1, y1, x2, y2 = (
                        round(item.bbox[0] * width),
                        round(item.bbox[1] * height),
                        round(item.bbox[2] * width),
                        round(item.bbox[3] * height),
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                    cv2.putText(
                        frame,
                        item.class_name,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 180, 0),
                        2,
                    )
                for role in roles:
                    x1, y1, x2, y2 = (
                        round(role["bbox"][0] * width),
                        round(role["bbox"][1] * height),
                        round(role["bbox"][2] * width),
                        round(role["bbox"][3] * height),
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 120, 0), 2)
                output_path = annotated_dir / path.name
                cv2.imwrite(str(output_path), frame)
                for event_index, event in enumerate(events):
                    event_id = (
                        f"{event.violation_type}-{path.stem}-"
                        f"{image_index:06d}-{event_index:03d}"
                    )
                    record = {
                        "event_id": event_id,
                        "violation_type": event.violation_type,
                        "rule_id": event.rule_id,
                        "source": str(path),
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "track_id": event.detection.track_id,
                        "vehicle_class": event.detection.class_name,
                        "plate_text": plates.get(event.detection.track_id),
                        "detection_confidence": event.detection.confidence,
                        "violation_confidence": event.confidence,
                        "evidence_image": str(output_path),
                        "details": {
                            **event.details,
                            "image_quality": quality.__dict__,
                        },
                    }
                    events_file.write(json.dumps(record) + "\n")
                    store.insert_event(record)
                    event_count += 1
    finally:
        store.close()
    print(f"Processed {len(images)} image(s); saved {event_count} event(s).")
    print(f"Annotated images: {annotated_dir}")


if __name__ == "__main__":
    main()
