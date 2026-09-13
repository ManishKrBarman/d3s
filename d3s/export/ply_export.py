"""
Point cloud and mesh export utilities.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_point_cloud(
    points,
    output_path,
    colors=None,
    confidence=None,
    format="ply",
):
    """
    Export a point cloud to a file.

    Parameters
    ----------
    points : ndarray [N, 3]
    output_path : str or Path
    colors : ndarray [N, 3], optional
        RGB colors in 0-1 range.
    confidence : ndarray [N], optional
        If provided and colors is None, creates a grayscale
        colormap from confidence values.
    format : str
        Output format ("ply", "xyz", "csv").
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "ply":
        _export_ply(
            points, output_path, colors, confidence
        )
    elif format == "xyz":
        _export_xyz(points, output_path)
    elif format == "csv":
        _export_csv(
            points, output_path, colors, confidence
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(
        "Exported %d points to %s",
        len(points),
        output_path,
    )


def _export_ply(points, path, colors, confidence):
    """Export using Open3D for proper PLY format."""

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif confidence is not None:
        # Grayscale from confidence
        cmin = confidence.min()
        cmax = confidence.max()

        if cmax > cmin:
            intensity = (confidence - cmin) / (cmax - cmin)
        else:
            intensity = np.ones_like(confidence)

        gray = np.repeat(
            intensity[:, None], 3, axis=1
        )
        pcd.colors = o3d.utility.Vector3dVector(gray)

    o3d.io.write_point_cloud(str(path), pcd)


def _export_xyz(points, path):
    """Export as simple XYZ text file."""

    np.savetxt(
        str(path),
        points,
        fmt="%.6f",
        delimiter=" ",
    )


def _export_csv(points, path, colors, confidence):
    """Export as CSV with optional attributes."""

    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["x", "y", "z"]
        if colors is not None:
            header.extend(["r", "g", "b"])
        if confidence is not None:
            header.append("confidence")

        writer.writerow(header)

        for i in range(len(points)):
            row = list(points[i])

            if colors is not None:
                row.extend(list(colors[i]))
            if confidence is not None:
                row.append(float(confidence[i]))

            writer.writerow(row)
