"""
GPS-based georeferencing for MASt3R reconstruction output.

Transforms the MASt3R metric-scale reconstruction into
real-world coordinates using GPS data from the drone.
"""

import logging
from typing import Optional

import numpy as np

from d3s.data.gps import latlon_to_local_enu

logger = logging.getLogger(__name__)


def umeyama_similarity(X, Y):
    """
    Compute the Umeyama similarity transform.

    Finds scale s, rotation R, translation t such that:
        Y ~= s * R @ X + t

    Parameters
    ----------
    X : ndarray [N, 3]
        Source points (e.g., MASt3R camera centers).
    Y : ndarray [N, 3]
        Target points (e.g., GPS-derived local coordinates).

    Returns
    -------
    dict
        - scale: float
        - rotation: ndarray [3, 3]
        - translation: ndarray [3]
        - rmse: float
    """

    assert X.shape == Y.shape
    assert X.shape[0] >= 3

    n = len(X)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    xc = X - mu_x
    yc = Y - mu_y

    covariance = (yc.T @ xc) / n

    U, D, Vt = np.linalg.svd(covariance)

    S = np.eye(3)

    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    var_x = np.sum(xc ** 2) / n

    scale = np.trace(np.diag(D) @ S) / var_x

    t = mu_y - scale * R @ mu_x

    # Compute residual
    Y_pred = (scale * (R @ X.T)).T + t
    errors = np.linalg.norm(Y - Y_pred, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "scale": float(scale),
        "rotation": R,
        "translation": t,
        "rmse": rmse,
        "per_point_errors": errors,
    }


def georeference(
    reconstruction_result,
    gps_data,
    image_id_map=None,
    earth_radius=6371000.0,
):
    """
    Georeference a MASt3R reconstruction using GPS data.

    Finds the similarity transform that maps MASt3R camera
    centers to GPS-derived local ENU coordinates, then
    applies it to the full point cloud.

    Parameters
    ----------
    reconstruction_result : ReconstructionResult
        Output from MASt3RPipeline.reconstruct().
    gps_data : DataFrame
        GPS data with columns: imgid, lat, lon, alt.
    image_id_map : dict, optional
        Maps image filename -> imgid for GPS lookup.
        If None, tries to parse numeric IDs from filenames.
    earth_radius : float
        Earth radius for coordinate conversion.

    Returns
    -------
    dict
        - transform: the similarity transform parameters
        - georef_points: transformed point cloud [M, 3]
        - georef_centers: transformed camera centers [N, 3]
        - gps_enu: GPS positions in local ENU [N, 3]
    """

    from pathlib import Path

    # Build GPS correspondences
    mast3r_centers = []
    gps_enu_list = []
    matched_names = []

    for i, img_path in enumerate(
        reconstruction_result.image_paths
    ):

        name = Path(img_path).stem

        # Determine image ID for GPS lookup
        if image_id_map is not None:
            img_id = image_id_map.get(name)
        else:
            try:
                img_id = int(name)
            except ValueError:
                continue

        if img_id is None:
            continue

        # Look up GPS
        gps_row = gps_data[gps_data["imgid"] == img_id]

        if gps_row.empty:
            continue

        gps_row = gps_row.iloc[0]

        mast3r_centers.append(
            reconstruction_result.camera_centers[i]
        )
        gps_enu_list.append(
            (gps_row["lat"], gps_row["lon"], gps_row["alt"])
        )
        matched_names.append(name)

    if len(mast3r_centers) < 3:
        raise ValueError(
            f"Need at least 3 GPS correspondences, "
            f"found {len(mast3r_centers)}"
        )

    # Convert GPS to local ENU
    lats = [g[0] for g in gps_enu_list]
    lons = [g[1] for g in gps_enu_list]
    alts = [g[2] for g in gps_enu_list]

    east, north, up = latlon_to_local_enu(
        lats, lons, alts, earth_radius=earth_radius
    )

    gps_enu = np.column_stack([east, north, up])
    mast3r_centers = np.array(mast3r_centers)

    logger.info(
        "Georeferencing with %d GPS correspondences",
        len(mast3r_centers),
    )

    # Compute similarity transform
    transform = umeyama_similarity(
        mast3r_centers, gps_enu
    )

    logger.info(
        "Georeference RMSE: %.4f m (scale=%.4f)",
        transform["rmse"],
        transform["scale"],
    )

    # Apply transform to point cloud
    s = transform["scale"]
    R = transform["rotation"]
    t = transform["translation"]

    georef_points = (
        s * (R @ reconstruction_result.merged_points.T)
    ).T + t

    georef_centers = (
        s * (R @ reconstruction_result.camera_centers.T)
    ).T + t

    return {
        "transform": transform,
        "georef_points": georef_points,
        "georef_centers": georef_centers,
        "gps_enu": gps_enu,
        "matched_names": matched_names,
    }
