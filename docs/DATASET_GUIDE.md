# Traffic model dataset guide

The repository supplies training and validation tooling, but not copyrighted or
jurisdiction-specific traffic imagery. Build datasets from cameras representative
of the intended deployment.

## Required classes

- Driver/rider model: `driver`, `rider`
- Helmet model: `helmet`, `no_helmet`
- Seatbelt model: `seatbelt`, `no_seatbelt`
- Plate model: `license_plate`
- Optional signal model: `red`, `yellow`, `green`, `off`

Separate models are usually easier to train and maintain than one model with
classes at very different scales.

## Collection coverage

Include day, night, rain, glare, shadows, motion blur, compression artifacts,
crowded traffic, occlusion, different vehicle categories, varied clothing, and
local plate formats. Record camera IDs and environmental conditions so results
can be reported by subgroup.

Obtain permission to collect and retain imagery. Minimize personal data and
follow applicable privacy and traffic-enforcement rules.

## Annotation and splitting

Use YOLO detection labels:

```text
class_id center_x center_y width height
```

Coordinates are normalized to 0–1. Split by camera and recording session—not
random adjacent frames—to prevent nearly identical frames leaking into
validation. A starting split is 70% train, 15% validation, and 15% untouched
test.

## Quality gates

```bash
python dataset_tools.py validate --root datasets/safety --classes driver,rider,no_helmet,no_seatbelt,license_plate
```

Before accepting a model, report per-class Precision, Recall, F1, AP50,
AP50–95, confusion matrices, OCR exact-match accuracy, character error rate,
latency, and subgroup results for weather, lighting, camera, and object size.

Store every model with a version, training command, dataset version, metrics,
and SHA-256 checksum in the model manifest.
