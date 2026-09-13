"""
GPS and coordinate transform utilities.

Handles loading GPS/IMU data and converting between:
  - WGS84 (lat/lon/alt)
  - Local ENU (East-North-Up) in meters
  - UTM coordinates
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default Earth radius (meters)
EARTH_RADIUS = 6371000.0


def latlon_to_local_enu(
    lat,
    lon,
    alt=None,
    lat_ref=None,
    lon_ref=None,
    alt_ref=None,
    earth_radius=EARTH_RADIUS,
):
    """
    Convert WGS84 lat/lon(/alt) to local East-North-Up
    coordinates in meters.

    Parameters
    ----------
    lat, lon : array-like
        Latitude and longitude in degrees.
    alt : array-like, optional
        Altitude in meters (MSL). If None, altitude is ignored.
    lat_ref, lon_ref : float, optional
        Reference point for the local frame. If None, uses
        the first point in the input arrays.
    alt_ref : float, optional
        Reference altitude. If None, uses the first altitude.
    earth_radius : float
        Approximate Earth radius in meters.

    Returns
    -------
    east, north, up : numpy arrays
        Local coordinates in meters.
    """

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if lat_ref is None:
        lat_ref = lat.flat[0]
    if lon_ref is None:
        lon_ref = lon.flat[0]

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    east = earth_radius * (lon_rad - lon_ref_rad) * np.cos(
        lat_ref_rad
    )
    north = earth_radius * (lat_rad - lat_ref_rad)

    if alt is not None:
        alt = np.asarray(alt, dtype=np.float64)
        if alt_ref is None:
            alt_ref = alt.flat[0]
        up = alt - alt_ref
    else:
        up = np.zeros_like(east)

    return east, north, up


def compute_motion(
    east,
    north,
    up=None,
):
    """
    Compute inter-frame motion from local ENU coordinates.

    Parameters
    ----------
    east, north : array-like
        Local coordinates in meters.
    up : array-like, optional
        Vertical coordinates in meters.

    Returns
    -------
    dict
        horizontal_motion : array of horizontal distances (m)
        vertical_motion : array of vertical distances (m)
        motion_3d : array of 3D distances (m)
        total_distance : total path length (m)
    """

    east = np.asarray(east)
    north = np.asarray(north)

    de = np.diff(east)
    dn = np.diff(north)
    horizontal = np.sqrt(de ** 2 + dn ** 2)

    if up is not None:
        up = np.asarray(up)
        du = np.abs(np.diff(up))
        motion_3d = np.sqrt(horizontal ** 2 + du ** 2)
    else:
        du = np.zeros_like(horizontal)
        motion_3d = horizontal

    return {
        "horizontal_motion": horizontal,
        "vertical_motion": du,
        "motion_3d": motion_3d,
        "total_distance": float(motion_3d.sum()),
    }


def load_gps_from_csv(
    csv_path,
    lat_col="lat",
    lon_col="lon",
    alt_col="alt",
    id_col="imgid",
):
    """
    Load GPS data from a CSV file.

    Returns a DataFrame with cleaned column names.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"GPS file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Validate required columns
    for col in [id_col, lat_col, lon_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in "
                f"{csv_path}. Available: {list(df.columns)}"
            )

    return df
