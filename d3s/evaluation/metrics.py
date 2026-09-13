"""
Evaluation metrics for 3D reconstruction quality.

Computes:
  - Reprojection error
  - Camera pose accuracy (vs ground truth)
  - Point cloud completeness
  - Nearest-neighbor distances between clouds
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def nearest_neighbor_distances(
    source,
    target,
    max_samples=50000,
):
    """
    Compute nearest-neighbor distances from source to target.

    Parameters
    ----------
    source : ndarray [N, 3]
        Source point cloud.
    target : ndarray [M, 3]
        Target point cloud to find neighbors in.
    max_samples : int
        Maximum source points to evaluate.

    Returns
    -------
    dict
        Percentile distances and statistics.
    """

    from scipy.spatial import cKDTree

    # Subsample if needed
    if len(source) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(
            len(source), size=max_samples, replace=False
        )
        source = source[idx]

    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)

    return {
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p50": float(np.percentile(distances, 50)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "mean": float(distances.mean()),
        "max": float(distances.max()),
        "num_evaluated": len(distances),
    }


def camera_pose_accuracy(
    estimated_centers,
    ground_truth_centers,
):
    """
    Compute camera pose accuracy metrics.

    Parameters
    ----------
    estimated_centers : ndarray [N, 3]
    ground_truth_centers : ndarray [N, 3]

    Returns
    -------
    dict
        Position error statistics.
    """

    errors = np.linalg.norm(
        estimated_centers - ground_truth_centers, axis=1
    )

    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "min": float(errors.min()),
        "per_image_errors": errors,
    }


def rotation_angle(R):
    """
    Compute the rotation angle (degrees) of a rotation matrix.
    """

    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def trajectory_accuracy(
    estimated_centers,
    ground_truth_centers,
):
    """
    Compute trajectory-level accuracy metrics including
    normalized RMSE.
    """

    errors = np.linalg.norm(
        estimated_centers - ground_truth_centers, axis=1
    )

    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Trajectory length
    gt_dists = np.linalg.norm(
        ground_truth_centers[1:]
        - ground_truth_centers[:-1],
        axis=1,
    )
    trajectory_length = float(gt_dists.sum())

    normalized_rmse = (
        rmse / trajectory_length
        if trajectory_length > 0
        else float("inf")
    )

    return {
        "rmse": rmse,
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "max_error": float(errors.max()),
        "trajectory_length": trajectory_length,
        "normalized_rmse": normalized_rmse,
    }
