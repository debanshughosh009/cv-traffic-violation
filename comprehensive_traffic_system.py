#!/usr/bin/env python3
"""Integrated traffic review pipeline for seven violation categories."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from advanced_preprocessing import enhance_adverse_frame
from evidence_store import EvidenceStore
from model_registry import canonical_model_label, validate_model_spec
from platform_features import (
    DirectionEngine,
    DirectionRule,
    PlatformEvent,
    TrackedObject,
    normalize_plate_text,
    parse_detection_label,
    point_in_polygon,
)
from two_wheeler_rider_count import Detection as RiderDetection
from two_wheeler_rider_count import associate_riders

TARGET_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--helmet-model", default=None)
    parser.add_argument("--seatbelt-model", default=None)
    parser.add_argument("--driver-model", default=None)
    parser.add_argument("--plate-model", default=None)
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/platform"))
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def preprocess_frame(cv2, frame, options):
    if not options.get("enabled", False):
        return frame, None
    return enhance_adverse_frame(cv2, frame, options)


def manifest_models(path):
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    failures = []
    for spec in raw.get("models", []):
        result = validate_model_spec(spec, path.parent)
        if result["ready"]:
            output[str(spec["task"])] = str(result["weights"])
        else:
            failures.append(f"{spec.get('task')}: {', '.join(result['issues'])}")
    if failures:
        raise SystemExit("Invalid model manifest:\n" + "\n".join(failures))
    return output


def line_side(point, p1, p2):
    cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (
        p2[1] - p1[1]
    ) * (point[0] - p1[0])
    return 1 if cross > 1e-9 else -1 if cross < -1e-9 else 0


class SceneRuleEngine:
    def __init__(self, config):
        self.stop_lines = tuple(config.get("stop_lines", []))
        self.parking_zones = tuple(config.get("parking_zones", []))
        self.previous: dict[int, tuple[float, float]] = {}
        self.parking: dict[tuple[str, int], tuple[float, tuple[float, float]]] = {}
        self.emitted: set[tuple[str, str, int]] = set()

    def update(self, detections, timestamp_ms, light_states):
        events = []
        for detection in detections:
            if detection.track_id is None:
                continue
            track = detection.track_id
            anchor = detection.anchor
            previous = self.previous.get(track)
            if previous:
                for rule in self.stop_lines:
                    old = line_side(previous, tuple(rule["p1"]), tuple(rule["p2"]))
                    new = line_side(anchor, tuple(rule["p1"]), tuple(rule["p2"]))
                    prohibited = int(rule.get("prohibited_side", 1))
                    if old == -prohibited and new == prohibited:
                        events.extend(
                            self._crossing(rule, detection, timestamp_ms, light_states)
                        )
            self.previous[track] = anchor
            events.extend(self._parking(detection, timestamp_ms))
        return events

    def _crossing(self, rule, detection, timestamp_ms, light_states):
        events = []
        key = ("stop_line", str(rule["id"]), detection.track_id)
        if key not in self.emitted:
            self.emitted.add(key)
            events.append(
                PlatformEvent(
                    "stop_line",
                    str(rule["id"]),
                    detection,
                    timestamp_ms,
                    detection.confidence,
                    {},
                )
            )
        light_id = rule.get("traffic_light_id")
        red_key = ("red_light", str(rule["id"]), detection.track_id)
        if light_id and light_states.get(light_id) == "red" and red_key not in self.emitted:
            self.emitted.add(red_key)
            events.append(
                PlatformEvent(
                    "red_light",
                    str(rule["id"]),
                    detection,
                    timestamp_ms,
                    detection.confidence,
                    {"traffic_light_id": light_id},
                )
            )
        return events

    def _parking(self, detection, timestamp_ms):
        events = []
        assert detection.track_id is not None
        for rule in self.parking_zones:
            rule_id = str(rule["id"])
            key = (rule_id, detection.track_id)
            polygon = tuple(tuple(point) for point in rule["polygon"])
            if not point_in_polygon(detection.anchor, polygon):
                self.parking.pop(key, None)
                continue
            state = self.parking.get(key)
            if state is None:
                self.parking[key] = (timestamp_ms, detection.anchor)
                continue
            entered, reference = state
            movement = math.dist(reference, detection.anchor)
            if movement > float(rule.get("max_movement", 0.02)):
                self.parking[key] = (timestamp_ms, detection.anchor)
                continue
            dwell = (timestamp_ms - entered) / 1000.0
            event_key = ("illegal_parking", rule_id, detection.track_id)
            if dwell >= float(rule.get("dwell_seconds", 10)) and event_key not in self.emitted:
                self.emitted.add(event_key)
                events.append(
                    PlatformEvent(
                        "illegal_parking",
                        rule_id,
                        detection,
                        timestamp_ms,
                        detection.confidence,
                        {"dwell_seconds": round(dwell, 2)},
                    )
                )
        return events


def light_states(cv2, frame, rules):
    height, width = frame.shape[:2]
    states = {}
    for rule in rules:
        x1, y1, x2, y2 = rule["roi"]
        crop = frame[
            round(y1 * height) : round(y2 * height),
            round(x1 * width) : round(x2 * width),
        ]
        if crop.size == 0:
            states[rule["id"]] = "unknown"
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        first = cv2.inRange(hsv, (0, 90, 80), (12, 255, 255))
        second = cv2.inRange(hsv, (165, 90, 80), (179, 255, 255))
        ratio = cv2.countNonZero(cv2.bitwise_or(first, second)) / first.size
        states[rule["id"]] = (
            "red" if ratio >= float(rule.get("red_ratio_threshold", 0.08)) else "not_red"
        )
    return states


def tracked_objects(result, names, width, height):
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        output.append(
            TrackedObject(
                names[int(box.cls[0])].lower(),
                float(box.conf[0]),
                (x1 / width, y1 / height, x2 / width, y2 / height),
                int(box.id[0]) if box.id is not None else None,
            )
        )
    return output


def attribute_events(model, frame, timestamp_ms, vehicles, device):
    if model is None:
        return []
    result = model.predict(frame, conf=0.2, device=device, verbose=False)[0]
    events = []
    height, width = frame.shape[:2]
    for box in result.boxes:
        violation = parse_detection_label(model.names[int(box.cls[0])])
        if not violation:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        candidates = [
            vehicle
            for vehicle in vehicles
            if vehicle.bbox[0] <= center[0] <= vehicle.bbox[2]
            and vehicle.bbox[1] <= center[1] <= vehicle.bbox[3]
        ]
        vehicle = candidates[0] if candidates else TrackedObject(
            "unknown", float(box.conf[0]), (x1 / width, y1 / height, x2 / width, y2 / height), None
        )
        events.append(
            PlatformEvent(
                violation,
                "specialized_model",
                vehicle,
                timestamp_ms,
                float(box.conf[0]),
                {"attribute_bbox": [x1 / width, y1 / height, x2 / width, y2 / height]},
            )
        )
    return events


def role_detections(model, frame, device):
    if model is None:
        return []
    result = model.predict(frame, conf=0.2, device=device, verbose=False)[0]
    height, width = frame.shape[:2]
    roles = []
    for box in result.boxes:
        canonical = canonical_model_label(model.names[int(box.cls[0])])
        if canonical not in ("driver", "rider"):
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        roles.append(
            {
                "role": canonical,
                "confidence": float(box.conf[0]),
                "bbox": (x1 / width, y1 / height, x2 / width, y2 / height),
            }
        )
    return roles


def plate_texts(model, frame, vehicles, ocr_backend, device):
    if model is None:
        return {}
    result = model.predict(frame, conf=0.2, device=device, verbose=False)[0]
    height, width = frame.shape[:2]
    texts = {}
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        owner = next(
            (
                vehicle
                for vehicle in vehicles
                if vehicle.track_id is not None
                and vehicle.bbox[0] <= center[0] <= vehicle.bbox[2]
                and vehicle.bbox[1] <= center[1] <= vehicle.bbox[3]
            ),
            None,
        )
        if owner is None or ocr_backend == "none":
            continue
        crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if crop.size and ocr_backend == "tesseract":
            try:
                import pytesseract
            except ImportError as exc:
                raise SystemExit("Install pytesseract and the Tesseract executable.") from exc
            text = normalize_plate_text(
                pytesseract.image_to_string(crop, config="--psm 7")
            )
            if len(text) >= 4:
                texts[owner.track_id] = text
    return texts


def triple_riding_events(objects, width, height, timestamp_ms):
    people = [
        RiderDetection(item.class_name, item.confidence, tuple(v * (width if i % 2 == 0 else height) for i, v in enumerate(item.bbox)), item.track_id)
        for item in objects
        if item.class_name == "person"
    ]
    bikes = [
        RiderDetection(item.class_name, item.confidence, tuple(v * (width if i % 2 == 0 else height) for i, v in enumerate(item.bbox)), item.track_id)
        for item in objects
        if item.class_name in ("bicycle", "motorcycle")
    ]
    estimates = associate_riders(people, bikes, width, height)
    events = []
    for estimate in estimates:
        if estimate.rider_count < 3:
            continue
        vehicle = next(
            (
                item
                for item in objects
                if item.track_id == estimate.vehicle.track_id
                and item.class_name == estimate.vehicle.class_name
            ),
            None,
        )
        if vehicle:
            events.append(
                PlatformEvent(
                    "triple_riding",
                    "rider_association",
                    vehicle,
                    timestamp_ms,
                    estimate.candidate_confidence or vehicle.confidence,
                    {"rider_count": estimate.rider_count},
                )
            )
    return events


def make_record(event, source, frame_index, evidence_path, plate):
    track = event.detection.track_id
    return {
        "event_id": f"{event.violation_type}-{event.rule_id}-{track}-{frame_index:08d}",
        "violation_type": event.violation_type,
        "rule_id": event.rule_id,
        "source": source,
        "frame_index": frame_index,
        "timestamp_ms": round(event.timestamp_ms, 2),
        "track_id": track,
        "vehicle_class": event.detection.class_name,
        "plate_text": plate,
        "detection_confidence": round(event.detection.confidence, 4),
        "violation_confidence": round(event.confidence, 4),
        "evidence_image": evidence_path.as_posix(),
        "details": event.details,
    }


def run():
    args = parse_args()
    try:
        import cv2
        from ultralytics import YOLO
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Install dependencies from requirements.txt") from exc
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = manifest_models(args.model_manifest)
    base_model = YOLO(args.model)
    helmet_path = args.helmet_model or manifest.get("helmet")
    seatbelt_path = args.seatbelt_model or manifest.get("seatbelt")
    driver_path = args.driver_model or manifest.get("driver")
    plate_path = args.plate_model or manifest.get("license_plate")
    helmet_model = YOLO(helmet_path) if helmet_path else None
    seatbelt_model = YOLO(seatbelt_path) if seatbelt_path else None
    driver_model = YOLO(driver_path) if driver_path else None
    plate_model = YOLO(plate_path) if plate_path else None
    names = dict(base_model.names)
    ids = [index for index, name in names.items() if name.lower() in TARGET_CLASSES]
    direction_rules = tuple(
        DirectionRule(
            str(rule["id"]),
            tuple(tuple(point) for point in rule["polygon"]),
            tuple(rule["allowed_vector"]),
            float(rule.get("min_displacement", 0.015)),
            int(rule.get("confirmation_frames", 3)),
        )
        for rule in config.get("direction_zones", [])
    )
    direction_engine = DirectionEngine(direction_rules)
    scene_engine = SceneRuleEngine(config)
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    events_file = (args.output_dir / "events.jsonl").open("w", encoding="utf-8")
    store = EvidenceStore(args.output_dir / "events.db")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_index = 0
    emitted: set[tuple[str, int | None]] = set()
    latencies = []
    counts = Counter()
    quality_counts = Counter()
    progress = tqdm(desc="Traffic platform", unit="frame")
    try:
        while args.max_frames == 0 or frame_index < args.max_frames:
            ok, original = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            frame, quality = preprocess_frame(
                cv2, original, config.get("preprocessing", {})
            )
            if quality:
                quality_counts["low_light"] += int(quality.low_light)
                quality_counts["motion_blur"] += int(quality.motion_blur)
                quality_counts["low_contrast"] += int(quality.low_contrast)
            height, width = frame.shape[:2]
            timestamp_ms = frame_index * 1000.0 / fps
            result = base_model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=ids,
                conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
            )[0]
            objects = tracked_objects(result, names, width, height)
            vehicles = [item for item in objects if item.class_name != "person"]
            lights = light_states(cv2, frame, config.get("traffic_lights", []))
            events = scene_engine.update(vehicles, timestamp_ms, lights)
            events += direction_engine.update(vehicles, frame_index, timestamp_ms)
            events += triple_riding_events(objects, width, height, timestamp_ms)
            events += attribute_events(helmet_model, frame, timestamp_ms, vehicles, args.device)
            events += attribute_events(seatbelt_model, frame, timestamp_ms, vehicles, args.device)
            roles = role_detections(driver_model, frame, args.device)
            plates = plate_texts(plate_model, frame, vehicles, args.ocr, args.device)
            for item in vehicles:
                x1, y1, x2, y2 = (
                    round(item.bbox[0] * width), round(item.bbox[1] * height),
                    round(item.bbox[2] * width), round(item.bbox[3] * height),
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(frame, f"{item.class_name} #{item.track_id}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 0), 2)
            for role in roles:
                x1, y1, x2, y2 = (
                    round(role["bbox"][0] * width),
                    round(role["bbox"][1] * height),
                    round(role["bbox"][2] * width),
                    round(role["bbox"][3] * height),
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 120, 0), 2)
                cv2.putText(
                    frame,
                    f"{role['role']} {role['confidence']:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 120, 0),
                    2,
                )
            for event in events:
                dedupe = (event.violation_type, event.detection.track_id)
                if event.detection.track_id is not None and dedupe in emitted:
                    continue
                emitted.add(dedupe)
                name = f"{event.violation_type}-{event.detection.track_id}-{frame_index:08d}.jpg"
                relative = Path("evidence") / name
                cv2.putText(frame, event.violation_type.upper(), (20, 40 + 28 * (counts[event.violation_type] % 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.imwrite(str(evidence_dir / name), frame)
                record = make_record(
                    event, args.source, frame_index, relative,
                    plates.get(event.detection.track_id),
                )
                events_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                events_file.flush()
                store.insert_event(record)
                counts[event.violation_type] += 1
            if args.output_video:
                if writer is None:
                    args.output_video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height)
                    )
                writer.write(frame)
            if not args.no_show:
                cv2.imshow("Traffic Violation Platform", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            latencies.append((time.perf_counter() - started) * 1000)
            frame_index += 1
            progress.update(1)
            progress.set_postfix(events=sum(counts.values()), refresh=False)
    finally:
        progress.close()
        capture.release()
        events_file.close()
        store.close()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    metrics = {
        "frames": frame_index,
        "events_by_type": dict(counts),
        "quality_flags": dict(quality_counts),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "throughput_fps": round(1000 / (sum(latencies) / len(latencies)), 2) if latencies else 0,
    }
    (args.output_dir / "runtime_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
