import unittest
from pathlib import Path

from two_wheeler_rider_count import (
    Detection,
    EventDeduplicator,
    RiderEstimate,
    associate_riders,
    bbox_iou,
    event_id,
    make_event_record,
    rider_count_bucket,
    triple_riding_confidence,
)


def detection(class_name, bbox, confidence=0.9, track_id=None):
    return Detection(class_name, confidence, bbox, track_id)


class RiderCountBucketTests(unittest.TestCase):
    def test_expected_buckets(self):
        self.assertEqual(rider_count_bucket(0), "unknown")
        self.assertEqual(rider_count_bucket(1), "1")
        self.assertEqual(rider_count_bucket(2), "2")
        self.assertEqual(rider_count_bucket(3), "3+")
        self.assertEqual(rider_count_bucket(5), "3+")

    def test_negative_count_is_invalid(self):
        with self.assertRaises(ValueError):
            rider_count_bucket(-1)


class RiderAssociationTests(unittest.TestCase):
    def test_empty_frame(self):
        self.assertEqual(associate_riders([], [], 640, 360), [])

    def test_person_above_vehicle_is_associated(self):
        bike = detection("motorcycle", (100, 200, 180, 260), track_id=3)
        rider = detection("person", (115, 80, 165, 225))
        estimates = associate_riders([rider], [bike], 640, 360)
        self.assertEqual(estimates[0].riders, (rider,))

    def test_nearby_pedestrian_outside_region_is_ignored(self):
        bike = detection("bicycle", (100, 200, 180, 260))
        pedestrian = detection("person", (300, 80, 350, 230))
        estimates = associate_riders([pedestrian], [bike], 640, 360)
        self.assertEqual(estimates[0].rider_count, 0)

    def test_person_is_assigned_to_only_nearest_of_two_bikes(self):
        left_bike = detection("motorcycle", (100, 200, 160, 250))
        right_bike = detection("motorcycle", (180, 200, 240, 250))
        rider = detection("person", (170, 90, 205, 225))
        estimates = associate_riders(
            [rider], [left_bike, right_bike], frame_width=640, frame_height=360
        )
        self.assertEqual([estimate.rider_count for estimate in estimates], [0, 1])


class CandidateConfidenceTests(unittest.TestCase):
    def test_requires_three_riders(self):
        bike = detection("motorcycle", (0, 0, 10, 10), confidence=0.8)
        riders = [detection("person", (0, 0, 1, 1), confidence=0.7)] * 2
        self.assertIsNone(triple_riding_confidence(bike, riders))

    def test_uses_vehicle_and_third_highest_rider_confidence(self):
        bike = detection("motorcycle", (0, 0, 10, 10), confidence=0.75)
        riders = [
            detection("person", (0, 0, 1, 1), confidence=value)
            for value in (0.95, 0.8, 0.6, 0.2)
        ]
        self.assertEqual(triple_riding_confidence(bike, riders), 0.6)

    def test_evidence_record_contains_required_fields(self):
        bike = detection(
            "motorcycle", (100, 200, 180, 260), confidence=0.75, track_id=4
        )
        riders = tuple(
            detection("person", (110 + offset, 80, 140 + offset, 225), confidence=0.8)
            for offset in (0, 10, 20)
        )
        record = make_event_record(
            "clip.mp4", 25, 1000.0, RiderEstimate(bike, riders), Path("evidence/event.jpg")
        )
        self.assertEqual(record["rider_count_bucket"], "3+")
        self.assertEqual(record["track_id"], 4)
        self.assertEqual(len(record["rider_boxes"]), 3)
        self.assertEqual(record["evidence_image"], "evidence/event.jpg")


class EventDeduplicationTests(unittest.TestCase):
    def test_repeated_track_emits_once(self):
        deduplicator = EventDeduplicator(max_age_frames=30)
        bike = detection("motorcycle", (100, 100, 180, 180), track_id=7)
        self.assertTrue(deduplicator.should_emit(bike, 0))
        self.assertFalse(deduplicator.should_emit(bike, 1))

    def test_missing_track_id_uses_spatial_fallback(self):
        deduplicator = EventDeduplicator(max_age_frames=30)
        first = detection("bicycle", (100, 100, 180, 180))
        moved = detection("bicycle", (105, 102, 185, 182))
        self.assertTrue(deduplicator.should_emit(first, 0))
        self.assertFalse(deduplicator.should_emit(moved, 1))

    def test_tracker_id_can_attach_to_prior_spatial_event(self):
        deduplicator = EventDeduplicator(max_age_frames=30)
        untracked = detection("bicycle", (100, 100, 180, 180))
        tracked = detection("bicycle", (105, 102, 185, 182), track_id=9)
        self.assertTrue(deduplicator.should_emit(untracked, 0))
        self.assertFalse(deduplicator.should_emit(tracked, 1))
        self.assertFalse(deduplicator.should_emit(tracked, 10))

    def test_spatial_fallback_expires(self):
        deduplicator = EventDeduplicator(max_age_frames=2)
        bike = detection("bicycle", (100, 100, 180, 180))
        self.assertTrue(deduplicator.should_emit(bike, 0))
        self.assertTrue(deduplicator.should_emit(bike, 3))

    def test_adjacent_distinct_bikes_can_both_emit(self):
        deduplicator = EventDeduplicator(max_age_frames=30)
        left = detection("motorcycle", (100, 100, 150, 180), track_id=1)
        right = detection("motorcycle", (160, 100, 210, 180), track_id=2)
        self.assertTrue(deduplicator.should_emit(left, 0))
        self.assertTrue(deduplicator.should_emit(right, 0))

    def test_bbox_iou_handles_non_overlapping_boxes(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_untracked_event_ids_include_position(self):
        left = detection("bicycle", (100, 100, 150, 180))
        right = detection("bicycle", (200, 100, 250, 180))
        self.assertNotEqual(event_id("clip.mp4", 3, left), event_id("clip.mp4", 3, right))


if __name__ == "__main__":
    unittest.main()
