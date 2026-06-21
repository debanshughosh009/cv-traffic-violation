import tempfile
import unittest
from pathlib import Path

from advanced_preprocessing import (
    FrameQuality,
    adaptive_preprocessing_options,
    classify_quality,
)
from analytics_report import build_time_series
from dataset_tools import (
    DatasetManifest,
    validate_dataset,
    write_yolo_dataset_template,
)
from model_registry import canonical_model_label, validate_model_spec
from performance_evaluation import (
    average_precision,
    character_error_rate,
    evaluate_events,
    evaluate_ocr,
)
from scalability_benchmark import summarize_samples


class AdvancedPreprocessingTests(unittest.TestCase):
    def test_classifies_dark_and_blurred_metrics(self):
        quality = classify_quality(brightness=15, blur_score=2, contrast=4)
        self.assertTrue(quality.low_light)
        self.assertTrue(quality.motion_blur)
        self.assertTrue(quality.low_contrast)

    def test_adaptive_options_enable_needed_corrections(self):
        options = adaptive_preprocessing_options(
            FrameQuality(
                brightness=20,
                blur_score=10,
                contrast=8,
                low_light=True,
                motion_blur=True,
                low_contrast=True,
            )
        )
        self.assertTrue(options["clahe"])
        self.assertTrue(options["shadow_correction"])
        self.assertTrue(options["deblur"])
        self.assertTrue(options["dehaze"])


class DatasetToolsTests(unittest.TestCase):
    def test_template_creates_split_directories_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yaml_path = write_yolo_dataset_template(
                root, ["driver", "no_helmet", "no_seatbelt", "license_plate"]
            )
            self.assertTrue(yaml_path.is_file())
            self.assertTrue((root / "images" / "train").is_dir())
            self.assertTrue((root / "labels" / "val").is_dir())

    def test_validator_reports_missing_labels_and_empty_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train" / "frame.jpg").write_bytes(b"image")
            manifest = DatasetManifest(root, ("driver",), ("train", "val"))
            result = validate_dataset(manifest)
            self.assertEqual(result["missing_labels"], 1)
            self.assertIn("val", result["empty_splits"])
            self.assertFalse(result["ready"])


class ModelRegistryTests(unittest.TestCase):
    def test_maps_driver_and_violation_labels(self):
        self.assertEqual(canonical_model_label("motorbike_driver"), "driver")
        self.assertEqual(canonical_model_label("without-helmet"), "helmet_non_compliance")

    def test_model_spec_rejects_missing_weights(self):
        result = validate_model_spec(
            {"task": "driver", "weights": "missing.pt", "required_labels": ["driver"]},
            Path("."),
        )
        self.assertFalse(result["ready"])
        self.assertIn("weights_not_found", result["issues"])


class TrendAnalyticsTests(unittest.TestCase):
    def test_builds_fixed_time_buckets_by_violation(self):
        events = [
            {"timestamp_ms": 1000, "violation_type": "wrong_side"},
            {"timestamp_ms": 59000, "violation_type": "wrong_side"},
            {"timestamp_ms": 61000, "violation_type": "red_light"},
        ]
        series = build_time_series(events, bucket_seconds=60)
        self.assertEqual(series[0]["total"], 2)
        self.assertEqual(series[0]["by_violation"]["wrong_side"], 2)
        self.assertEqual(series[1]["by_violation"]["red_light"], 1)


class FormalEvaluationTests(unittest.TestCase):
    def test_average_precision_rewards_ranked_true_positive(self):
        self.assertAlmostEqual(
            average_precision([(0.9, True), (0.8, False)], positives=1), 1.0
        )

    def test_evaluation_includes_accuracy_and_map(self):
        truth = [{"violation_type": "wrong_side", "source": "a", "frame_index": 10}]
        predictions = [
            {
                "violation_type": "wrong_side",
                "source": "a",
                "frame_index": 10,
                "violation_confidence": 0.9,
            }
        ]
        result = evaluate_events(truth, predictions)
        self.assertEqual(result["micro"]["accuracy"], 1.0)
        self.assertEqual(result["mAP"], 1.0)

    def test_ocr_evaluation_reports_exact_match_and_character_error(self):
        result = evaluate_ocr(
            [{"event_id": "a", "plate_text": "KA01AB1234"}],
            [{"event_id": "a", "plate_text": "KA01A81234"}],
        )
        self.assertEqual(result["exact_match_accuracy"], 0.0)
        self.assertGreater(result["character_error_rate"], 0.0)
        self.assertAlmostEqual(character_error_rate("ABC", "ADC"), 1 / 3, places=4)


class ScalabilityTests(unittest.TestCase):
    def test_summarizes_latency_throughput_and_failures(self):
        summary = summarize_samples(
            [
                {"latency_ms": 100.0, "ok": True, "memory_mb": 200.0},
                {"latency_ms": 200.0, "ok": True, "memory_mb": 250.0},
                {"latency_ms": 300.0, "ok": False, "memory_mb": 240.0},
            ],
            elapsed_seconds=1.0,
        )
        self.assertEqual(summary["successful_frames"], 2)
        self.assertEqual(summary["failed_frames"], 1)
        self.assertEqual(summary["throughput_fps"], 2.0)
        self.assertEqual(summary["p95_latency_ms"], 290.0)
        self.assertEqual(summary["peak_memory_mb"], 250.0)


if __name__ == "__main__":
    unittest.main()
