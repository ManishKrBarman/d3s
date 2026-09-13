"""
Image quality filtering.

Computes quality metrics (sharpness, brightness, contrast)
and filters out poor-quality frames before reconstruction.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_quality_metrics(image_path):
    """
    Compute quality metrics for a single image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - sharpness: Laplacian variance (higher = sharper)
        - brightness: mean grayscale intensity (0-255)
        - contrast: std dev of grayscale (higher = more contrast)
        - width, height: image dimensions
        Returns None if the image cannot be read.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness via Laplacian variance
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    # Brightness and contrast
    brightness = float(gray.mean())
    contrast = float(gray.std())

    height, width = image.shape[:2]

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "width": width,
        "height": height,
    }


def filter_by_quality(
    image_paths,
    min_sharpness=None,
    min_sharpness_percentile=None,
    min_brightness=30,
    max_brightness=240,
    min_contrast=15,
):
    """
    Filter a list of images by quality metrics.

    Parameters
    ----------
    image_paths : list of Path
        Images to evaluate.
    min_sharpness : float, optional
        Absolute minimum sharpness threshold.
    min_sharpness_percentile : int, optional
        Reject the bottom N% by sharpness. Used only if
        min_sharpness is None.
    min_brightness : float
        Minimum mean brightness.
    max_brightness : float
        Maximum mean brightness.
    min_contrast : float
        Minimum contrast (std dev).

    Returns
    -------
    dict
        - passed: list of (path, metrics) that passed
        - rejected: list of (path, metrics, reason) that failed
        - sharpness_threshold: the computed threshold
    """

    # Compute all metrics
    all_metrics = []

    for i, path in enumerate(image_paths, start=1):
        metrics = compute_quality_metrics(path)

        if metrics is not None:
            all_metrics.append((path, metrics))

        if i % 100 == 0:
            logger.info(
                "Quality analysis: %d/%d",
                i,
                len(image_paths),
            )

    if not all_metrics:
        return {
            "passed": [],
            "rejected": [],
            "sharpness_threshold": 0,
        }

    # Determine sharpness threshold
    if min_sharpness is not None:
        sharpness_threshold = min_sharpness
    elif min_sharpness_percentile is not None:
        sharpness_values = [
            m["sharpness"] for _, m in all_metrics
        ]
        sharpness_threshold = float(
            np.percentile(
                sharpness_values,
                min_sharpness_percentile,
            )
        )
    else:
        sharpness_threshold = 0

    logger.info(
        "Sharpness threshold: %.2f",
        sharpness_threshold,
    )

    # Apply filters
    passed = []
    rejected = []

    for path, metrics in all_metrics:

        # Check sharpness
        if metrics["sharpness"] < sharpness_threshold:
            rejected.append(
                (path, metrics, "low_sharpness")
            )
            continue

        # Check brightness
        if metrics["brightness"] < min_brightness:
            rejected.append(
                (path, metrics, "too_dark")
            )
            continue

        if metrics["brightness"] > max_brightness:
            rejected.append(
                (path, metrics, "too_bright")
            )
            continue

        # Check contrast
        if metrics["contrast"] < min_contrast:
            rejected.append(
                (path, metrics, "low_contrast")
            )
            continue

        passed.append((path, metrics))

    logger.info(
        "Quality filter: %d passed, %d rejected "
        "out of %d",
        len(passed),
        len(rejected),
        len(all_metrics),
    )

    return {
        "passed": passed,
        "rejected": rejected,
        "sharpness_threshold": sharpness_threshold,
    }
