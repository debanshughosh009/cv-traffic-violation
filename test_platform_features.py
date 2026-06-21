import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from analytics_report import build_summary
from evidence_store import EvidenceStore
from performance_evaluation import evaluate_events
from platform_features import (
    DirectionEngine,
    DirectionRule,
    TrackedObject,
    normalize_plate_text,
    parse_detection_label,
)


class DirectionEngineTests(unittest.TestCase):
    def test_emits_wrong_side_after_sustained_opposite_motion(self):
        rule = DirectionRule(
            "north_lane",
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            allowed_vector=(0.0, -1.0),
            min_displacement=0.05,
            confirmation_frames=2,
        )
        engine = DirectionEngine((rule,))
        vehicle = lambda y: TrackedObject("car", 0.9, (0.4, y - 0.1, 0.6, y), 7)
        self.assertEqual(engine.update([vehicle(0.3)], 0, 0.0), [])
        self.assertEqual(engine.update([vehicle(0.4)], 1, 100.0), [])
        events = engine.update([vehicle(0.5)], 2, 200.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].violation_type, "wrong_side")
        self.assertEqual(events[0].rule_id, "north_lane")

    def test_allowed_direction_does_not_emit(self):
        rule = DirectionRule(
            "north_lane",
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            allowed_vector=(0.0, -1.0),
            min_displacement=0.05,
            confirmation_frames=1,
        )
        engine = DirectionEngine((rule,))
        first = TrackedObject("car", 0.9, (0.4, 0.7, 0.6, 0.8), 2)
        second = TrackedObject("car", 0.9, (0.4, 0.5, 0.6, 0.6), 2)
        engine.update([first], 0, 0.0)
        self.assertEqual(engine.update([second], 1, 100.0), [])


class AttributeAndPlateTests(unittest.TestCase):
    def test_normalizes_indian_plate_text(self):
        self.assertEqual(normalize_plate_text(" ka-01 ab 1234\n"), "KA01AB1234")

    def test_parses_common_violation_labels(self):
        self.assertEqual(parse_detection_label("no-helmet"), "helmet_non_compliance")
        self.assertEqual(parse_detection_label("without_seatbelt"), "seatbelt_non_compliance")
        self.assertIsNone(parse_detection_label("helmet"))


class EvidenceStoreTests(unittest.TestCase):
    def test_insert_search_and_deduplicate_event(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "events.db")
            event = {
                "event_id": "event-1",
                "violation_type": "wrong_side",
                "rule_id": "lane-a",
                "source": "clip.mp4",
                "frame_index": 12,
                "timestamp_ms": 400.0,
                "track_id": 9,
                "vehicle_class": "car",
                "plate_text": "KA01AB1234",
                "violation_confidence": 0.8,
                "evidence_image": "evidence/event-1.jpg",
                "details": {"direction_similarity": -0.9},
            }
            self.assertTrue(store.insert_event(event))
            self.assertFalse(store.insert_event(event))
            rows = store.search(violation_type="wrong_side", plate_text="KA01")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["details"]["direction_similarity"], -0.9)
            store.close()


class AnalyticsTests(unittest.TestCase):
    def test_summary_groups_types_and_plates(self):
        events = [
            {
                "violation_type": "wrong_side",
                "timestamp_ms": 1000,
                "plate_text": "KA01AB1234",
                "violation_confidence": 0.8,
            },
            {
                "violation_type": "wrong_side",
                "timestamp_ms": 2000,
                "plate_text": "KA01AB1234",
                "violation_confidence": 0.6,
            },
            {
                "violation_type": "red_light",
                "timestamp_ms": 3000,
                "plate_text": None,
                "violation_confidence": 0.9,
            },
        ]
        summary = build_summary(events)
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["by_violation"]["wrong_side"], 2)
        self.assertEqual(summary["top_plates"][0], ["KA01AB1234", 2])
        self.assertAlmostEqual(summary["average_confidence"], 0.7667, places=4)


class EvaluationTests(unittest.TestCase):
    def test_computes_per_class_and_micro_metrics(self):
        truth = [
            {"violation_type": "wrong_side", "source": "a", "frame_index": 10},
            {"violation_type": "red_light", "source": "a", "frame_index": 20},
        ]
        predictions = [
            {
                "violation_type": "wrong_side",
                "source": "a",
                "frame_index": 11,
                "violation_confidence": 0.9,
            },
            {
                "violation_type": "illegal_parking",
                "source": "a",
                "frame_index": 30,
                "violation_confidence": 0.8,
            },
        ]
        result = evaluate_events(truth, predictions, frame_tolerance=2)
        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["fp"], 1)
        self.assertEqual(result["micro"]["fn"], 1)
        self.assertEqual(result["micro"]["precision"], 0.5)
        self.assertEqual(result["micro"]["recall"], 0.5)
        self.assertEqual(result["micro"]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
