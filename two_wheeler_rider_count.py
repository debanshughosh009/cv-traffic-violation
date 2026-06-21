#!/usr/bin/env python3
"""Estimate riders on two-wheelers in dashcam video and save 3+ candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TARGET_CLASSES = ("person", "bicycle", "motorcycle")
TWO_WHEELER_CLASSES = frozenset(("bicycle", "motorcycle"))
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A model detection normalized into plain Python values."""

    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None = None


@dataclass(frozen=True)
class RiderEstimate:
    """The people geometrically associated with one two-wheeler."""

    vehicle: Detection
    riders: tuple[Detection, ...]

    @property
    def rider_count(self) -> int:
        return len(self.riders)

    @property
    def bucket(self) -> str:
        return rider_count_bucket(self.rider_count)

    @property
    def candidate_confidence(self) -> float | None:
        return triple_riding_confidence(self.vehicle, self.riders)


@dataclass
class _RecentEvent:
    class_name: str
    bbox: BBox
    last_frame: int


class EventDeduplicator:
    """Emit once per tracker ID, with a short spatial fallback for missing IDs."""

    def __init__(self, max_age_frames: int, iou_threshold: float = 0.2) -> None:
        self.max_age_frames = max(1, max_age_frames)
        self.iou_threshold = iou_threshold
        self._emitted_track_keys: set[str] = set()
        self._events_by_track_key: dict[str, _RecentEvent] = {}
        self._recent_events: list[_RecentEvent] = []

    def should_emit(self, detection: Detection, frame_index: int) -> bool:
        track_key = (
            f"{detection.class_name}:{detection.track_id}"
            if detection.track_id is not None
            else None
        )
        if track_key is not None and track_key in self._emitted_track_keys:
            event = self._events_by_track_key[track_key]
            event.bbox = detection.bbox
            event.last_frame = frame_index
            return False

        self._recent_events = [
            event
            for event in self._recent_events
            if frame_index - event.last_frame <= self.max_age_frames
        ]
        for event in self._recent_events:
            if (
                event.class_name == detection.class_name
                and bbox_iou(event.bbox, detection.bbox) >= self.iou_threshold
            ):
                event.bbox = detection.bbox
                event.last_frame = frame_index
                if track_key is not None:
                    self._emitted_track_keys.add(track_key)
                    self._events_by_track_key[track_key] = event
                return False

        event = _RecentEvent(detection.class_name, detection.bbox, frame_index)
        if track_key is not None:
            self._emitted_track_keys.add(track_key)
            self._events_by_track_key[track_key] = event
        self._recent_events.append(event)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count riders on bicycles and motorcycles in dashcam footage and save "
            "single-frame 3+ rider review candidates."
        )
    )
    parser.add_argument("--source", required=True, help="Recorded video path or video source.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics-compatible YOLO weights. Defaults to yolov8n.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rider_count"),
        help="Directory for events.jsonl and evidence images.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path for an annotated video containing all rider estimates.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Zero processes the complete source.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Minimum inference confidence, including person detections.",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.20,
        help="Minimum confidence for bicycle and motorcycle detections.",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable the preview window.")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be zero or greater.")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")
    if not args.conf <= args.vehicle_conf <= 1.0:
        raise SystemExit("--vehicle-conf must be between --conf and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0 and 1.")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be greater than zero.")


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
    if source.isdigit() and not Path(source).exists():
        return int(source)
    return source


def normalize_model_names(model_names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(model_names, dict):
        return dict(model_names)
    return dict(enumerate(model_names))


def resolve_class_ids(model_names: dict[int, str] | list[str], requested: Iterable[str]) -> list[int]:
    names_by_id = normalize_model_names(model_names)
    ids_by_name = {name.lower(): class_id for class_id, name in names_by_id.items()}
    missing = sorted(set(requested) - set(ids_by_name))
    if missing:
        raise SystemExit(f"Model does not contain required classes: {', '.join(missing)}")
    return [ids_by_name[name] for name in requested]


def rider_count_bucket(count: int) -> str:
    if count < 0:
        raise ValueError("Rider count cannot be negative.")
    if count == 0:
        return "unknown"
    if count >= 3:
        return "3+"
    return str(count)


def bottom_center(bbox: BBox) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def association_region(vehicle_bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    """Return a permissive region above a two-wheeler where riders may appear."""
    x1, y1, x2, y2 = vehicle_bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return (
        max(0.0, x1 - 0.75 * width),
        max(0.0, y1 - 4.0 * height),
        min(float(frame_width), x2 + 0.75 * width),
        min(float(frame_height), y2 + 0.5 * height),
    )


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def normalized_association_distance(person: Detection, vehicle: Detection) -> float:
    person_x, person_y = bottom_center(person.bbox)
    x1, y1, x2, y2 = vehicle.bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    target_x = (x1 + x2) / 2.0
    target_y = (y1 + y2) / 2.0
    return math.hypot((person_x - target_x) / width, (person_y - target_y) / height)


def associate_riders(
    people: Sequence[Detection],
    vehicles: Sequence[Detection],
    frame_width: int,
    frame_height: int,
) -> list[RiderEstimate]:
    """Assign each person to at most one nearby two-wheeler."""
    assigned: list[list[Detection]] = [[] for _ in vehicles]
    regions = [association_region(vehicle.bbox, frame_width, frame_height) for vehicle in vehicles]

    for person in people:
        anchor = bottom_center(person.bbox)
        candidates = [
            index for index, region in enumerate(regions) if point_in_bbox(anchor, region)
        ]
        if not candidates:
            continue
        best_index = min(
            candidates,
            key=lambda index: (normalized_association_distance(person, vehicles[index]), index),
        )
        assigned[best_index].append(person)

    return [
        RiderEstimate(vehicle=vehicle, riders=tuple(assigned[index]))
        for index, vehicle in enumerate(vehicles)
    ]


def triple_riding_confidence(
    vehicle: Detection, riders: Sequence[Detection]
) -> float | None:
    if len(riders) < 3:
        return None
    third_highest_person_confidence = sorted(
        (rider.confidence for rider in riders), reverse=True
    )[2]
    return min(vehicle.confidence, third_highest_person_confidence)


def bbox_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def detections_from_result(
    result, names_by_id: dict[int, str], vehicle_confidence: float
) -> tuple[list[Detection], list[Detection]]:
    people: list[Detection] = []
    vehicles: list[Detection] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names_by_id[class_id].lower()
        confidence = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        detection = Detection(
            class_name=class_name,
            confidence=confidence,
            bbox=tuple(float(value) for value in box.xyxy[0].tolist()),
            track_id=track_id,
        )
        if class_name == "person":
            people.append(detection)
        elif class_name in TWO_WHEELER_CLASSES and confidence >= vehicle_confidence:
            vehicles.append(detection)
    return people, vehicles


def should_show_window(no_show: bool) -> bool:
    if no_show:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("No DISPLAY found; running without a preview window.", file=sys.stderr)
        return False
    return True


def usable_fps(raw_fps: float) -> float:
    return raw_fps if math.isfinite(raw_fps) and raw_fps > 0.0 else 30.0


def create_video_writer(cv2, output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), usable_fps(fps), (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video for writing: {output_path}")
    return writer


def integer_bbox(bbox: BBox) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in bbox)


def draw_box_and_label(cv2, frame, bbox: BBox, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = integer_bbox(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = max(y1, text_size[1] + baseline + 4)
    cv2.rectangle(
        frame,
        (x1, label_y - text_size[1] - baseline - 4),
        (x1 + text_size[0] + 8, label_y + baseline - 2),
        color,
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


def draw_estimates(cv2, frame, estimates: Sequence[RiderEstimate]) -> None:
    for estimate in estimates:
        is_candidate = estimate.rider_count >= 3
        vehicle_color = (0, 0, 220) if is_candidate else (0, 180, 255)
        confidence_suffix = (
            f" candidate {estimate.candidate_confidence:.2f}" if is_candidate else ""
        )
        label = (
            f"{estimate.vehicle.class_name} riders:{estimate.bucket}{confidence_suffix}"
        )
        draw_box_and_label(cv2, frame, estimate.vehicle.bbox, label, vehicle_color)
        for rider in estimate.riders:
            draw_box_and_label(
                cv2, frame, rider.bbox, f"rider {rider.confidence:.2f}", (220, 120, 0)
            )


def safe_source_stem(source: str) -> str:
    stem = Path(source).stem or "stream"
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")
    return normalized or "stream"


def event_id(source: str, frame_index: int, detection: Detection) -> str:
    if detection.track_id is not None:
        identity = f"track-{detection.track_id}"
    else:
        x1, y1, x2, y2 = detection.bbox
        identity = f"untracked-x{round((x1 + x2) / 2)}-y{round((y1 + y2) / 2)}"
    return (
        f"{safe_source_stem(source)}-{frame_index:08d}-"
        f"{detection.class_name}-{identity}"
    )


def bbox_json(bbox: BBox) -> list[float]:
    return [round(value, 2) for value in bbox]


def make_event_record(
    source: str,
    frame_index: int,
    timestamp_ms: float,
    estimate: RiderEstimate,
    evidence_path: Path,
) -> dict[str, object]:
    confidence = estimate.candidate_confidence
    if confidence is None:
        raise ValueError("Evidence records require at least three associated riders.")
    return {
        "event_id": event_id(source, frame_index, estimate.vehicle),
        "source": source,
        "frame_index": frame_index,
        "timestamp_ms": round(timestamp_ms, 2),
        "track_id": estimate.vehicle.track_id,
        "two_wheeler_class": estimate.vehicle.class_name,
        "two_wheeler_bbox": bbox_json(estimate.vehicle.bbox),
        "two_wheeler_confidence": round(estimate.vehicle.confidence, 4),
        "rider_count": estimate.rider_count,
        "rider_count_bucket": estimate.bucket,
        "rider_boxes": [
            {"bbox": bbox_json(rider.bbox), "confidence": round(rider.confidence, 4)}
            for rider in estimate.riders
        ],
        "candidate_confidence": round(confidence, 4),
        "evidence_image": evidence_path.as_posix(),
    }


def frame_timestamp_ms(capture, cv2, frame_index: int, fps: float) -> float:
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(timestamp) and (timestamp > 0.0 or frame_index == 0):
        return timestamp
    return frame_index * 1000.0 / usable_fps(fps)


def run() -> None:
    args = parse_args()
    cv2, YOLO, tqdm = import_dependencies()
    model = YOLO(args.model)
    names_by_id = normalize_model_names(model.names)
    class_ids = resolve_class_ids(names_by_id, TARGET_CLASSES)

    capture = cv2.VideoCapture(normalize_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    deduplicator = EventDeduplicator(max_age_frames=round(usable_fps(fps) * 2.0))
    writer = None
    show_window = should_show_window(args.no_show)
    processed_frames = 0
    candidate_count = 0
    raw_total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_total = (
        int(raw_total_frames)
        if math.isfinite(raw_total_frames) and raw_total_frames > 0
        else None
    )
    if args.max_frames > 0:
        total_frames = min(source_total, args.max_frames) if source_total else args.max_frames
    else:
        total_frames = source_total
    progress = tqdm(
        total=total_frames,
        desc="Counting riders",
        unit="frame",
        dynamic_ncols=True,
    )

    try:
        with events_path.open("w", encoding="utf-8") as events_file:
            while args.max_frames == 0 or processed_frames < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index = processed_frames
                frame_height, frame_width = frame.shape[:2]
                if args.output_video and writer is None:
                    writer = create_video_writer(
                        cv2, args.output_video, fps, frame_width, frame_height
                    )

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=class_ids,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                people, vehicles = detections_from_result(
                    result, names_by_id, args.vehicle_conf
                )
                estimates = associate_riders(
                    people, vehicles, frame_width=frame_width, frame_height=frame_height
                )
                draw_estimates(cv2, frame, estimates)

                timestamp_ms = frame_timestamp_ms(capture, cv2, frame_index, fps)
                for estimate in estimates:
                    if estimate.rider_count < 3:
                        continue
                    if not deduplicator.should_emit(estimate.vehicle, frame_index):
                        continue
                    identifier = event_id(args.source, frame_index, estimate.vehicle)
                    relative_evidence_path = Path("evidence") / f"{identifier}.jpg"
                    absolute_evidence_path = args.output_dir / relative_evidence_path
                    if not cv2.imwrite(str(absolute_evidence_path), frame):
                        raise SystemExit(
                            f"Could not write evidence image: {absolute_evidence_path}"
                        )
                    record = make_event_record(
                        args.source,
                        frame_index,
                        timestamp_ms,
                        estimate,
                        relative_evidence_path,
                    )
                    events_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    events_file.flush()
                    candidate_count += 1
                    progress.write(
                        f"Candidate {record['event_id']}: {estimate.rider_count} rider(s) "
                        f"at {timestamp_ms / 1000.0:.2f}s"
                    )

                if writer:
                    writer.write(frame)
                progress.update(1)
                progress.set_postfix(
                    two_wheelers=len(estimates), candidates=candidate_count, refresh=False
                )
                if show_window:
                    cv2.imshow("Two-Wheeler Rider Count", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        processed_frames += 1
                        break

                processed_frames += 1
    finally:
        progress.close()
        capture.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print(
        f"Finished. Processed {processed_frames} frame(s); "
        f"saved {candidate_count} candidate event(s)."
    )
    print(f"Event metadata: {events_path}")
    if args.output_video:
        print(f"Annotated video: {args.output_video}")


if __name__ == "__main__":
    run()
