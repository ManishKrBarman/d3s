"""
Adaptive keyframe selection from image sequences.

Merges the logic from select_keyframes.py (v1, quality-score based)
and select_keyframes_v2.py (geometric inlier-ratio based) into a
single configurable module.

Two strategies:
  - "adaptive": feature-matching based (ORB + fundamental matrix)
  - "uniform": simple every-N-th frame selection
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from d3s.config import KeyframeConfig

logger = logging.getLogger(__name__)


def select_keyframes(
    image_paths,
    config=None,
    quality_metrics=None,
):
    """
    Select keyframes from a sequence of images.

    Parameters
    ----------
    image_paths : list of Path
        Ordered list of image file paths.
    config : KeyframeConfig, optional
        Configuration. Uses defaults if None.
    quality_metrics : dict, optional
        Pre-computed quality metrics keyed by path.
        If None, sharpness is computed on the fly.

    Returns
    -------
    dict
        - selected: list of selected image paths
        - selection_records: list of dicts with selection details
        - stats: summary statistics
    """

    if config is None:
        config = KeyframeConfig()

    if config.strategy == "uniform":
        return _select_uniform(image_paths, config)

    return _select_adaptive(
        image_paths, config, quality_metrics
    )


# ==============================================================
# Uniform selection
# ==============================================================


def _select_uniform(image_paths, config):
    """Select every N-th frame."""

    interval = config.uniform_interval

    selected = image_paths[::interval]

    # Always include first and last
    if image_paths[0] not in selected:
        selected.insert(0, image_paths[0])

    if image_paths[-1] not in selected:
        selected.append(image_paths[-1])

    return {
        "selected": selected,
        "selection_records": [
            {
                "path": str(p),
                "index": image_paths.index(p),
                "reason": "uniform",
            }
            for p in selected
        ],
        "stats": {
            "total_frames": len(image_paths),
            "selected_frames": len(selected),
            "interval": interval,
        },
    }


# ==============================================================
# Adaptive selection
# ==============================================================


def _extract_features(image_path, orb):
    """Extract ORB keypoints and descriptors."""

    image = cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    keypoints, descriptors = orb.detectAndCompute(
        image, None
    )

    return keypoints, descriptors


def _compute_inlier_ratio(
    kp1, desc1, kp2, desc2, matcher, config
):
    """
    Compute geometric inlier ratio between two sets
    of features using fundamental matrix estimation.
    """

    if desc1 is None or desc2 is None:
        return 0.0

    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [
        m
        for m in matches
        if m.distance < config.match_distance
    ]

    if len(good_matches) < 8:
        return 0.0

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    _, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        config.ransac_threshold,
        config.ransac_confidence,
    )

    if mask is None:
        return 0.0

    inliers = int(mask.sum())

    return inliers / len(good_matches)


def _compute_sharpness(image_path):
    """Compute Laplacian variance sharpness."""

    image = cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return 0.0

    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _select_adaptive(image_paths, config, quality_metrics):
    """
    Adaptive keyframe selection using ORB features and
    geometric inlier ratio.
    """

    if len(image_paths) < 2:
        return {
            "selected": list(image_paths),
            "selection_records": [],
            "stats": {"total_frames": len(image_paths)},
        }

    # Compute sharpness for all frames
    logger.info("Computing sharpness for %d frames...",
                len(image_paths))

    sharpness = {}

    for path in image_paths:
        if quality_metrics and path in quality_metrics:
            sharpness[path] = quality_metrics[path][
                "sharpness"
            ]
        else:
            sharpness[path] = _compute_sharpness(path)

    # Determine sharpness threshold
    sharp_values = list(sharpness.values())
    sharpness_threshold = float(
        np.percentile(
            sharp_values,
            config.min_sharpness_percentile,
        )
    )

    logger.info(
        "Sharpness threshold (P%d): %.2f",
        config.min_sharpness_percentile,
        sharpness_threshold,
    )

    # Initialize ORB and matcher
    orb = cv2.ORB_create(nfeatures=config.n_features)

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING, crossCheck=True
    )

    # Pre-extract features
    logger.info("Extracting ORB features...")
    features = {}

    for i, path in enumerate(image_paths, start=1):

        kp, desc = _extract_features(path, orb)
        features[path] = (kp, desc)

        if i % 50 == 0:
            logger.info(
                "Features: %d/%d", i, len(image_paths)
            )

    # Selection loop
    selected = [image_paths[0]]  # Always keep first
    records = [
        {
            "path": str(image_paths[0]),
            "index": 0,
            "gap": 0,
            "inlier_ratio": None,
            "sharpness": sharpness[image_paths[0]],
            "reason": "first_frame",
        }
    ]

    last_idx = 0

    for idx in range(1, len(image_paths)):

        path = image_paths[idx]
        gap = idx - last_idx
        sharp = sharpness[path]

        # Skip extremely blurry frames (unless gap is too big)
        if (
            sharp < sharpness_threshold
            and gap < config.max_frame_gap
        ):
            continue

        # Compare with last selected frame
        kp1, desc1 = features[selected[-1]]
        kp2, desc2 = features[path]

        inlier_ratio = _compute_inlier_ratio(
            kp1, desc1, kp2, desc2, matcher, config
        )

        should_select = False
        reason = ""

        # Rule 1: Maximum gap exceeded
        if gap >= config.max_frame_gap:
            should_select = True
            reason = "max_gap"

        # Rule 2: Significant viewpoint change
        elif (
            gap >= config.min_frame_gap
            and inlier_ratio <= config.low_redundancy
            and sharp >= sharpness_threshold
        ):
            should_select = True
            reason = "large_viewpoint_change"

        # Rule 3: Moderate viewpoint change
        elif (
            gap >= config.min_frame_gap
            and config.low_redundancy
            < inlier_ratio
            < config.high_redundancy
            and sharp >= sharpness_threshold
        ):
            should_select = True
            reason = "moderate_viewpoint_change"

        if should_select:
            selected.append(path)
            records.append(
                {
                    "path": str(path),
                    "index": idx,
                    "gap": gap,
                    "inlier_ratio": inlier_ratio,
                    "sharpness": sharp,
                    "reason": reason,
                }
            )
            last_idx = idx

    # Always include final frame
    if selected[-1] != image_paths[-1]:
        path = image_paths[-1]
        gap = len(image_paths) - 1 - last_idx

        selected.append(path)
        records.append(
            {
                "path": str(path),
                "index": len(image_paths) - 1,
                "gap": gap,
                "inlier_ratio": None,
                "sharpness": sharpness[path],
                "reason": "last_frame",
            }
        )

    # Summary stats
    gaps = [r["gap"] for r in records if r["gap"] > 0]

    stats = {
        "total_frames": len(image_paths),
        "selected_frames": len(selected),
        "reduction_pct": round(
            100 * (1 - len(selected) / len(image_paths)),
            1,
        ),
        "min_gap": int(min(gaps)) if gaps else 0,
        "mean_gap": round(
            float(np.mean(gaps)), 1
        ) if gaps else 0,
        "max_gap": int(max(gaps)) if gaps else 0,
        "sharpness_threshold": sharpness_threshold,
    }

    logger.info(
        "Selected %d/%d frames (%.1f%% reduction)",
        stats["selected_frames"],
        stats["total_frames"],
        stats["reduction_pct"],
    )

    return {
        "selected": selected,
        "selection_records": records,
        "stats": stats,
    }
