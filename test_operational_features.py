import json
import tempfile
import unittest
from pathlib import Path

from evidence_store import EvidenceStore
from image_processor import discover_images
from security_controls import AuditLogger, hash_token, token_matches
from training_pipeline import build_training_command


class SecurityTests(unittest.TestCase):
    def test_token_hash_round_trip(self):
        encoded = hash_token("secret-token", salt=b"0123456789abcdef")
        self.assertTrue(token_matches("secret-token", encoded))
        self.assertFalse(token_matches("wrong-token", encoded))

    def test_audit_logger_writes_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            logger = AuditLogger(path)
            logger.write("review.updated", actor="analyst", target="event-1")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["action"], "review.updated")
            self.assertEqual(record["actor"], "analyst")


class HumanReviewTests(unittest.TestCase):
    def test_review_status_can_be_updated_and_searched(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "events.db")
            store.insert_event(
                {
                    "event_id": "event-1",
                    "violation_type": "wrong_side",
                    "source": "clip.mp4",
                    "frame_index": 10,
                    "timestamp_ms": 100,
                    "violation_confidence": 0.8,
                    "details": {},
                }
            )
            self.assertTrue(
                store.update_review(
                    "event-1", "confirmed", reviewer="alice", notes="clear evidence"
                )
            )
            rows = store.search(review_status="confirmed")
            self.assertEqual(rows[0]["review_status"], "confirmed")
            self.assertEqual(rows[0]["reviewer"], "alice")
            store.close()

    def test_invalid_review_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "events.db")
            with self.assertRaises(ValueError):
                store.update_review("missing", "maybe")
            store.close()


class TrainingPipelineTests(unittest.TestCase):
    def test_builds_reproducible_yolo_training_command(self):
        command = build_training_command(
            data=Path("data/helmet/dataset.yaml"),
            base_model="yolov8n.pt",
            epochs=50,
            image_size=640,
            project=Path("runs/training"),
            name="helmet-v1",
            seed=42,
        )
        self.assertEqual(command[:3], ["yolo", "detect", "train"])
        self.assertIn("seed=42", command)
        self.assertIn("name=helmet-v1", command)


class ImageWorkflowTests(unittest.TestCase):
    def test_discovers_supported_images_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"")
            (root / "nested" / "b.PNG").write_bytes(b"")
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            self.assertEqual(
                [path.name for path in discover_images([root])], ["a.jpg", "b.PNG"]
            )


if __name__ == "__main__":
    unittest.main()
