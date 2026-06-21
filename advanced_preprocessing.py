"""Adaptive image-quality assessment and adverse-condition preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    blur_score: float
    contrast: float
    low_light: bool
    motion_blur: bool
    low_contrast: bool


def classify_quality(
    brightness: float,
    blur_score: float,
    contrast: float,
    low_light_threshold: float = 55.0,
    blur_threshold: float = 45.0,
    contrast_threshold: float = 18.0,
) -> FrameQuality:
    return FrameQuality(
        brightness=round(float(brightness), 3),
        blur_score=round(float(blur_score), 3),
        contrast=round(float(contrast), 3),
        low_light=brightness < low_light_threshold,
        motion_blur=blur_score < blur_threshold,
        low_contrast=contrast < contrast_threshold,
    )


def adaptive_preprocessing_options(quality: FrameQuality) -> dict[str, bool]:
    return {
        "clahe": quality.low_light or quality.low_contrast,
        "shadow_correction": quality.low_light,
        "deblur": quality.motion_blur,
        "dehaze": quality.low_contrast,
        "rain_suppression": quality.low_contrast and quality.motion_blur,
    }


def assess_frame_quality(cv2, frame, options: dict | None = None) -> FrameQuality:
    options = options or {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return classify_quality(
        brightness,
        blur_score,
        contrast,
        float(options.get("low_light_threshold", 55.0)),
        float(options.get("blur_threshold", 45.0)),
        float(options.get("contrast_threshold", 18.0)),
    )


def enhance_adverse_frame(cv2, frame, options: dict) -> tuple[object, FrameQuality]:
    """Apply conservative adaptive enhancement and return quality metadata."""
    quality = assess_frame_quality(cv2, frame, options)
    selected = adaptive_preprocessing_options(quality) if options.get(
        "adaptive", True
    ) else {
        name: bool(options.get(name, False))
        for name in (
            "clahe",
            "shadow_correction",
            "deblur",
            "dehaze",
            "rain_suppression",
        )
    }
    output = frame.copy()
    if selected["rain_suppression"] and options.get("rain_suppression", True):
        median = cv2.medianBlur(output, 3)
        output = cv2.addWeighted(output, 0.65, median, 0.35, 0)
    if selected["shadow_correction"] and options.get("shadow_correction", True):
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        value = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(value)
        output = cv2.cvtColor(
            cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR
        )
    if selected["clahe"] and options.get("clahe", True):
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = cv2.createCLAHE(2.0, (8, 8)).apply(lightness)
        output = cv2.cvtColor(
            cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR
        )
    if selected["dehaze"] and options.get("dehaze", True):
        output = cv2.detailEnhance(output, sigma_s=10, sigma_r=0.15)
    if selected["deblur"] and options.get("deblur", True):
        blurred = cv2.GaussianBlur(output, (0, 0), 1.0)
        output = cv2.addWeighted(output, 1.8, blurred, -0.8, 0)
    return output, quality
