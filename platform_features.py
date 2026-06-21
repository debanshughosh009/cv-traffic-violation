"""Reusable traffic-platform logic with no heavy computer-vision dependencies."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-9
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class TrackedObject:
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None

    @property
    def anchor(self) -> Point:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


@dataclass(frozen=True)
class DirectionRule:
    rule_id: str
    polygon: tuple[Point, ...]
    allowed_vector: Point
    min_displacement: float = 0.02
    confirmation_frames: int = 3


@dataclass(frozen=True)
class PlatformEvent:
    violation_type: str
    rule_id: str
    detection: TrackedObject
    timestamp_ms: float
    confidence: float
    details: dict[str, object]


@dataclass
class _DirectionState:
    anchor: Point
    opposite_frames: int
    last_frame: int


class DirectionEngine:
    """Detect sustained travel opposite to a calibrated lane direction."""

    def __init__(self, rules: Sequence[DirectionRule], stale_frames: int = 150) -> None:
        self.rules = tuple(rules)
        self.stale_frames = max(1, stale_frames)
        self._states: dict[tuple[str, int], _DirectionState] = {}
        self._emitted: set[tuple[str, int]] = set()

    def update(
        self,
        detections: Sequence[TrackedObject],
        frame_index: int,
        timestamp_ms: float,
    ) -> list[PlatformEvent]:
        events: list[PlatformEvent] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            for rule in self.rules:
                key = (rule.rule_id, detection.track_id)
                if not point_in_polygon(detection.anchor, rule.polygon):
                    self._states.pop(key, None)
                    continue
                state = self._states.get(key)
                if state is None:
                    self._states[key] = _DirectionState(
                        detection.anchor, 0, frame_index
                    )
                    continue
                dx = detection.anchor[0] - state.anchor[0]
                dy = detection.anchor[1] - state.anchor[1]
                displacement = math.hypot(dx, dy)
                if displacement >= rule.min_displacement:
                    allowed_length = math.hypot(*rule.allowed_vector)
                    if allowed_length <= 1e-9:
                        raise ValueError(f"Direction rule {rule.rule_id} has a zero vector.")
                    similarity = (
                        dx * rule.allowed_vector[0] + dy * rule.allowed_vector[1]
                    ) / (displacement * allowed_length)
                    state.opposite_frames = (
                        state.opposite_frames + 1 if similarity < 0.0 else 0
                    )
                    state.anchor = detection.anchor
                    if (
                        state.opposite_frames >= rule.confirmation_frames
                        and key not in self._emitted
                    ):
                        self._emitted.add(key)
                        events.append(
                            PlatformEvent(
                                "wrong_side",
                                rule.rule_id,
                                detection,
                                timestamp_ms,
                                min(detection.confidence, abs(similarity)),
                                {
                                    "direction_similarity": round(similarity, 4),
                                    "confirmation_frames": state.opposite_frames,
                                },
                            )
                        )
                state.last_frame = frame_index
        stale = [
            key
            for key, state in self._states.items()
            if frame_index - state.last_frame > self.stale_frames
        ]
        for key in stale:
            del self._states[key]
        return events


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output while preserving only registration-like characters."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_detection_label(label: str) -> str | None:
    """Map common custom-model class labels to canonical violation types."""
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    no_helmet = {"no_helmet", "without_helmet", "helmetless", "nohelmet"}
    no_seatbelt = {
        "no_seatbelt",
        "without_seatbelt",
        "seatbelt_missing",
        "noseatbelt",
    }
    if normalized in no_helmet:
        return "helmet_non_compliance"
    if normalized in no_seatbelt:
        return "seatbelt_non_compliance"
    return None
