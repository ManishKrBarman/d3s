"""
Georeferenced output export (placeholder for GeoTIFF support).

For now, exports georeferenced point clouds with metadata.
Full GeoTIFF DSM/DTM generation is a future enhancement.
"""

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_georef_metadata(
    georef_result,
    output_dir,
):
    """
    Save georeferencing metadata and transformed data.

    Parameters
    ----------
    georef_result : dict
        Output from d3s.reconstruction.georef.georeference().
    output_dir : str or Path
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = georef_result["transform"]

    # Save transform parameters
    np.savez(
        output_dir / "georef_transform.npz",
        scale=transform["scale"],
        rotation=transform["rotation"],
        translation=transform["translation"],
    )

    # Save georeferenced camera centers
    centers_path = output_dir / "georef_camera_centers.csv"

    with open(centers_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "east_m", "north_m", "up_m"])

        for name, center in zip(
            georef_result["matched_names"],
            georef_result["georef_centers"],
        ):
            writer.writerow(
                [
                    name,
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                ]
            )

    logger.info(
        "Georeferencing metadata saved to %s "
        "(RMSE=%.4f m, scale=%.4f)",
        output_dir,
        transform["rmse"],
        transform["scale"],
    )

    return {
        "transform_file": output_dir / "georef_transform.npz",
        "centers_file": centers_path,
    }
