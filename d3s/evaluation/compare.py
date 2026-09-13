"""
Comparison utilities for evaluating reconstruction
against ground truth or reference data.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from d3s.evaluation.metrics import (
    camera_pose_accuracy,
    nearest_neighbor_distances,
    trajectory_accuracy,
)

logger = logging.getLogger(__name__)


def compare_with_ground_truth(
    reconstruction_result,
    ground_truth_df,
    output_dir=None,
):
    """
    Compare reconstruction camera poses against
    ground truth positions.

    Parameters
    ----------
    reconstruction_result : ReconstructionResult
        Pipeline output.
    ground_truth_df : DataFrame
        Ground truth with columns:
        imgid, x_gt, y_gt, z_gt
    output_dir : Path, optional
        Directory to save comparison results.

    Returns
    -------
    dict
        Comparison metrics.
    """

    from d3s.reconstruction.georef import (
        umeyama_similarity,
    )

    # Match reconstruction images to ground truth
    est_centers = []
    gt_centers = []

    for i, img_path in enumerate(
        reconstruction_result.image_paths
    ):
        name = Path(img_path).stem

        try:
            img_id = int(name)
        except ValueError:
            continue

        gt_row = ground_truth_df[
            ground_truth_df["imgid"] == img_id
        ]

        if gt_row.empty:
            continue

        gt_row = gt_row.iloc[0]

        est_centers.append(
            reconstruction_result.camera_centers[i]
        )

        gt_centers.append(
            [
                float(gt_row["x_gt"]),
                float(gt_row["y_gt"]),
                float(gt_row["z_gt"]),
            ]
        )

    if len(est_centers) < 3:
        logger.warning(
            "Only %d ground truth correspondences found",
            len(est_centers),
        )
        return None

    est_centers = np.array(est_centers)
    gt_centers = np.array(gt_centers)

    # Align with similarity transform first
    transform = umeyama_similarity(est_centers, gt_centers)

    s = transform["scale"]
    R = transform["rotation"]
    t = transform["translation"]

    aligned_centers = (s * (R @ est_centers.T)).T + t

    # Compute metrics
    pose_accuracy = camera_pose_accuracy(
        aligned_centers, gt_centers
    )

    traj_accuracy = trajectory_accuracy(
        aligned_centers, gt_centers
    )

    result = {
        "num_correspondences": len(est_centers),
        "alignment_scale": transform["scale"],
        "alignment_rmse": transform["rmse"],
        "pose_accuracy": pose_accuracy,
        "trajectory_accuracy": traj_accuracy,
    }

    # Save if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _save_comparison(
            output_dir, result, gt_centers, aligned_centers
        )

    return result


def _save_comparison(
    output_dir, result, gt_centers, aligned_centers
):
    """Save comparison results to files."""

    # Summary text
    summary_path = output_dir / "comparison_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=== RECONSTRUCTION vs GROUND TRUTH ===\n\n")

        ta = result["trajectory_accuracy"]

        f.write(f"Correspondences: {result['num_correspondences']}\n")
        f.write(f"Alignment scale: {result['alignment_scale']:.6f}\n")
        f.write(f"RMSE:            {ta['rmse']:.6f} m\n")
        f.write(f"Mean error:      {ta['mean_error']:.6f} m\n")
        f.write(f"Median error:    {ta['median_error']:.6f} m\n")
        f.write(f"Max error:       {ta['max_error']:.6f} m\n")
        f.write(f"Trajectory len:  {ta['trajectory_length']:.2f} m\n")
        f.write(f"Normalized RMSE: {ta['normalized_rmse']:.6f}\n")

    logger.info("Comparison saved to %s", summary_path)
