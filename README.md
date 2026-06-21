# cv_traffic_violation

An extensible YOLO/OpenCV traffic-violation review platform for recorded video,
webcams, and fixed-camera streams.

## Feature status

| Feature | Status | Remaining work |
|---|---|---|
| Low-light enhancement | Added | CLAHE and gamma correction are available |
| Rain handling | Added as adaptive preprocessing | Validate against real rain footage |
| Shadow handling | Added as adaptive preprocessing | Validate thresholds per camera |
| Motion-blur handling | Added as quality filtering and sharpening | Severe blur still requires better capture or learned restoration |
| Input normalization | Added | Quality assessment selects corrections per frame |
| Vehicle detection and classification | Added | Uses YOLO vehicle classes |
| Pedestrian detection | Added | Uses the YOLO `person` class |
| Rider detection | Partially added | Replace geometric association with a dedicated rider model |
| Driver detection | Integrated | Supply and validate driver/rider model weights |
| Helmet non-compliance | Partially added | Supply and validate custom helmet weights and training data |
| Seatbelt non-compliance | Partially added | Supply and validate custom seatbelt weights and training data |
| Triple riding | Added as prototype | Improve and evaluate rider association accuracy |
| Wrong-side driving | Added | Calibrate lane polygons and allowed directions per camera |
| Stop-line violation | Added | Calibrate stop-line coordinates per camera |
| Red-light violation | Added | Improve signal-state recognition beyond color thresholding |
| Illegal parking | Added | Add legal-exception and congestion handling |
| Violation classification | Added | Uses predefined event categories |
| Confidence scores | Added | Stores detection and violation confidence |
| Number-plate detection | Partially added | Supply and validate custom plate-detection weights |
| Registration OCR | Partially added | Install Tesseract and tune OCR for local plate formats |
| Annotated evidence images | Added | Evidence images are generated |
| Metadata and timestamps | Added | Stored in JSONL and SQLite |
| Violation statistics | Added | Counts and average confidence are generated |
| Trend analytics | Added | Includes time buckets, source comparisons, and review summaries |
| Searchable records | Added | SQLite search supports type, plate, and source filters |
| Summary reports | Added | JSON, CSV, and HTML reports are generated |
| Precision, Recall, and F1 | Added | Event-level evaluation is available |
| Accuracy | Added at event level | Supply labelled evaluation records for meaningful results |
| mAP | Added at event level | Detection-model mAP still requires labelled YOLO validation data |
| Computational efficiency | Added at basic level | Runtime latency and throughput are recorded |
| Scalability evaluation | Added | Run it on target hardware and production stream counts |

The remaining work is primarily data and validation: collect representative
labelled images, train specialized models, calibrate each camera, and execute
accuracy and scalability tests on the intended deployment hardware.

The generic COCO model cannot recognize helmets, seatbelts, or license plates.
Those features are integrated, but require task-specific YOLO weights.

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For OCR, also install the native
[Tesseract executable](https://github.com/tesseract-ocr/tesseract). The Python
package alone is not the OCR engine.

The default model is `yolov8n.pt`. Ultralytics downloads it automatically on
the first run. If the machine has no internet access, download the weights
separately and pass the local path with `--model`.

## Run

Use a webcam:

```bash
python3 vehicle_yolo_stream.py --source 0
```

Use a video file:

```bash
python3 vehicle_yolo_stream.py --source path/to/video.mp4
```

Use an RTSP/HTTP stream:

```bash
python3 vehicle_yolo_stream.py --source rtsp://user:password@camera-host:554/stream1
```

Save an annotated output video:

```bash
python3 vehicle_yolo_stream.py --source path/to/video.mp4 --output runs/vehicles.mp4 --no-show
```

Press `q` or `Esc` to stop the preview window.

Both video processors show a `tqdm` progress bar with frame throughput and live
detection counts. Recorded files include completion percentage and ETA; streams
with no known length show an open-ended frame counter.

## Options

- `--vehicle-classes`: comma-separated COCO classes to detect. Defaults to
  `car,motorcycle,bus,truck`. Add bicycles with
  `--vehicle-classes bicycle,car,motorcycle,bus,truck`.
- `--conf`: confidence threshold. Defaults to `0.35`.
- `--device`: inference device such as `cpu`, `cuda`, `cuda:0`, or `mps`.
- `--model`: any Ultralytics-compatible YOLO weights file.

## Complete traffic platform

Calibrate a copy of `comprehensive_rules.example.json`, then run:

```bash
python comprehensive_traffic_system.py --source test_video.mp4 --config comprehensive_rules.example.json --output-dir runs/platform --output-video runs/platform/annotated.mp4 --no-show
```

Enable specialized models when weights are available:

```bash
python comprehensive_traffic_system.py --source test_video.mp4 --config comprehensive_rules.example.json --driver-model models/driver.pt --helmet-model models/helmet.pt --seatbelt-model models/seatbelt.pt --plate-model models/license_plate.pt --ocr tesseract --output-dir runs/platform --no-show
```

Helmet weights should expose a class such as `no_helmet` or
`without_helmet`. Seatbelt weights should expose `no_seatbelt` or
`without_seatbelt`. The plate model should localize registration plates.
Driver weights should expose `driver` and/or `rider`.

Models can instead be supplied through `model_manifest.example.json`:

```bash
python model_registry.py --manifest model_manifest.example.json
python comprehensive_traffic_system.py --source test_video.mp4 --config comprehensive_rules.example.json --model-manifest model_manifest.example.json --ocr tesseract --output-dir runs/platform --no-show
```

### Configuration

All points use normalized image coordinates from `[0, 0]` at the top-left to
`[1, 1]` at the bottom-right.

- `preprocessing` controls adaptive low-light, shadow, haze/rain, and blur
  processing. Quality flags are recorded in `runtime_metrics.json`.
- `stop_lines` defines directed crossing lines and optional linked signals.
- `traffic_lights` defines red-signal regions of interest.
- `parking_zones` defines polygons, dwell times and movement tolerances.
- `direction_zones` defines lane polygons and allowed movement vectors.

The output folder contains:

- `events.jsonl` for portable event metadata.
- `events.db` for indexed SQLite search.
- `evidence/` with annotated event images.
- `runtime_metrics.json` with mean latency and throughput.
- The optional annotated output video.

### Search stored events

```bash
python search_events.py --database runs/platform/events.db --type wrong_side
python search_events.py --database runs/platform/events.db --plate KA01
python search_events.py --database runs/platform/events.db --review-status confirmed
```

### Process photographs and image folders

```bash
python image_processor.py images/ --config comprehensive_rules.example.json --model-manifest model_manifest.example.json --ocr tesseract --output-dir runs/images
```

Single photographs support road-user and driver detection, triple-riding
candidates, helmet and seatbelt models, and plate OCR. Motion-dependent rules
such as wrong-side travel and illegal parking require video.

### Create and train specialized datasets

```bash
python dataset_tools.py create --root datasets/safety --classes driver,rider,no_helmet,no_seatbelt,license_plate
python dataset_tools.py validate --root datasets/safety --classes driver,rider,no_helmet,no_seatbelt,license_plate
python training_pipeline.py --data datasets/safety/dataset.yaml --name safety-v1 --epochs 100 --seed 42
```

Add `--export onnx` to train, validate, and export an ONNX model. See
`docs/DATASET_GUIDE.md` for annotation and split requirements.

### Analytics and reports

```bash
python analytics_report.py --database runs/platform/events.db --output-dir runs/reports
```

This creates `summary.json`, `events.csv`, and a self-contained `report.html`.

### Formal event evaluation

Create labelled ground truth as JSON or JSONL:

```json
{"violation_type":"wrong_side","source":"test_video.mp4","frame_index":120}
```

Then evaluate predictions:

```bash
python performance_evaluation.py --ground-truth labels.jsonl --predictions runs/platform/events.jsonl --frame-tolerance 5 --output runs/evaluation.json
```

The report contains per-class and micro true positives, false positives, false
negatives, Accuracy, Precision, Recall, F1, average precision, and event mAP.
Add `--evaluate-ocr` to report exact plate accuracy and character error rate.
Detection-model mAP should also be calculated with Ultralytics validation:

```bash
yolo detect val model=models/helmet.pt data=data/helmet.yaml
```

### Human review, audit, and retention

```bash
python review_events.py --database runs/platform/events.db list --status pending
python review_events.py --database runs/platform/events.db update --event-id EVENT_ID --status confirmed --reviewer analyst@example.com --notes "Clear evidence"
python retention_control.py --database runs/platform/events.db --evidence-root runs/platform --days 90
python retention_control.py --database runs/platform/events.db --evidence-root runs/platform --days 90 --apply
```

Review and retention changes are recorded in an append-only JSONL audit log.
The retention command defaults to a dry run.

### Authenticated API and Docker deployment

Generate a password-derived API token hash:

```bash
python -c "from security_controls import hash_token; print(hash_token('replace-me'))"
```

Set it as `TRAFFIC_API_TOKEN_HASH`, then run:

```bash
python deployment_server.py --database runs/platform/events.db --host 0.0.0.0 --port 8080
```

Endpoints are `GET /health`, authenticated `GET /events`, `GET /summary`, and
`POST /events/{event_id}/review`. A `Dockerfile` is included.

### Scalability benchmark

```bash
python scalability_benchmark.py --workers 2 --repetitions 2 --output runs/scalability.json -- python comprehensive_traffic_system.py --source test_video.mp4 --config comprehensive_rules.example.json --max-frames 100 --output-dir runs/benchmark --no-show
```

The report includes successful and failed jobs, throughput, mean latency, p95,
p99, and peak memory when workers provide memory samples.

## Count riders on two-wheelers

`two_wheeler_rider_count.py` detects people, bicycles, and motorcycles, then
associates each person with at most one nearby two-wheeler. It labels rider
counts as `unknown`, `1`, `2`, or `3+`. The first `3+` observation for a tracked
two-wheeler is saved as a review candidate.

Process a recorded dashcam video:

```bash
python3 two_wheeler_rider_count.py \
  --source test_video.mp4 \
  --output-dir runs/rider_count \
  --no-show
```

Run a short smoke test and save an annotated video:

```bash
python3 two_wheeler_rider_count.py \
  --source test_video.mp4 \
  --max-frames 300 \
  --output-video runs/rider_count/annotated.mp4 \
  --no-show
```

The defaults prioritize recall on small dashcam objects: inference confidence
`0.15`, two-wheeler confidence `0.20`, and image size `960`. Use `--conf`,
`--vehicle-conf`, and `--imgsz` to tune them.

### Evidence output

The output directory contains:

- `events.jsonl`: one JSON object for each deduplicated `3+` candidate.
- `evidence/`: the corresponding annotated images.
- An annotated video when `--output-video` is provided.

Each JSONL record includes the event ID, source, frame index, video timestamp,
track ID, two-wheeler class/box/confidence, rider count and boxes, candidate
confidence, and relative evidence-image path. `events.jsonl` is created even
when a run finds no candidates.

### Limitations

Rider association is a permissive geometry heuristic over generic COCO
detections. It can confuse nearby pedestrians, miss occluded riders, and perform
poorly on distant or blurred two-wheelers. A `3+` result is evidence for human
review, not an enforcement decision. Accuracy evaluation requires dedicated
positive and negative dashcam clips; the included video is only a smoke-test
source.

## Tests

The association, confidence, bucketing, and deduplication logic uses only the
Python standard library and can be tested without loading YOLO:

```bash
python3 -m unittest -v
```

## Important limitations

This is a configurable review prototype, not an enforcement-grade system.
Stop-line, red-light, parking, and direction rules require a stable camera and
site-specific calibration. Color-based signal detection can fail under glare,
rain, reflections, or nighttime conditions. Illegal-parking logic cannot infer
legal exceptions. Helmet, seatbelt, and plate quality depends primarily on the
specialized datasets and weights supplied by the operator. Every event requires
human review and jurisdiction-specific validation.
