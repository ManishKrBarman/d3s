"""
Unified dataset abstraction for the D3S pipeline.

Supports two input modes:
  1. Pre-extracted image directories (e.g., AGZ Zurich dataset)
  2. Raw video files (MP4/MOV) — delegates to video_ingest

Provides a clean interface for accessing images, GPS data,
IMU data, and ground truth regardless of the source format.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Dataset:
    """
    A dataset of geo-tagged aerial images.

    Parameters
    ----------
    root : str or Path
        Root directory of the dataset.
    image_subdir : str
        Subdirectory containing images (relative to root).
    log_subdir : str
        Subdirectory containing log/CSV files (relative to root).
    image_ext : str
        Image file extension to look for.
    """

    def __init__(
        self,
        root,
        image_subdir="MAV Images",
        log_subdir="Log Files",
        image_ext=".jpg",
    ):
        self.root = Path(root)
        self.image_dir = self.root / image_subdir
        self.log_dir = self.root / log_subdir
        self.image_ext = image_ext

        # Loaded data (lazy)
        self._gps = None
        self._pose = None
        self._ground_truth = None
        self._image_index = None

    # ----------------------------------------------------------
    # Image discovery
    # ----------------------------------------------------------

    @property
    def image_index(self):
        """
        Sorted list of (image_id, image_path) tuples
        for all images found on disk.
        """

        if self._image_index is None:
            self._image_index = self._build_image_index()

        return self._image_index

    @property
    def image_ids(self):
        """List of image IDs found on disk."""
        return [entry[0] for entry in self.image_index]

    @property
    def image_paths(self):
        """List of image paths found on disk."""
        return [entry[1] for entry in self.image_index]

    @property
    def num_images(self):
        return len(self.image_index)

    def _build_image_index(self):
        """
        Scan the image directory and build a sorted index.
        Attempts to parse numeric image IDs from filenames.
        """

        if not self.image_dir.exists():
            logger.warning(
                "Image directory does not exist: %s",
                self.image_dir,
            )
            return []

        entries = []

        for path in self.image_dir.glob(f"*{self.image_ext}"):
            try:
                image_id = int(path.stem)
                entries.append((image_id, path))
            except ValueError:
                # Non-numeric filename — use hash as ID
                image_id = hash(path.stem) & 0x7FFFFFFF
                entries.append((image_id, path))

        entries.sort(key=lambda x: x[0])

        return entries

    def get_image_path(self, image_id):
        """Return the path for a specific image ID."""

        for entry_id, path in self.image_index:
            if entry_id == image_id:
                return path

        raise FileNotFoundError(
            f"Image ID {image_id} not found in {self.image_dir}"
        )

    # ----------------------------------------------------------
    # GPS data
    # ----------------------------------------------------------

    @property
    def gps(self):
        """GPS DataFrame, loaded lazily."""

        if self._gps is None:
            self._gps = self._load_gps()

        return self._gps

    def _load_gps(self):
        """Load GPS data from OnboardGPS.csv."""

        gps_path = self.log_dir / "OnboardGPS.csv"

        if not gps_path.exists():
            logger.warning("No GPS file found: %s", gps_path)
            return pd.DataFrame()

        df = pd.read_csv(gps_path)
        df.columns = df.columns.str.strip()

        return df

    def get_gps(self, image_id):
        """Get GPS record for a specific image."""

        if self.gps.empty:
            return None

        result = self.gps[self.gps["imgid"] == image_id]

        if result.empty:
            return None

        return result.iloc[0]

    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    @property
    def ground_truth(self):
        """Ground truth DataFrame, loaded lazily."""

        if self._ground_truth is None:
            self._ground_truth = self._load_ground_truth()

        return self._ground_truth

    def _load_ground_truth(self):
        """Load ground truth from GroundTruthAGL.csv."""

        gt_path = self.log_dir / "GroundTruthAGL.csv"

        if not gt_path.exists():
            logger.warning(
                "No ground truth file found: %s", gt_path
            )
            return pd.DataFrame()

        df = pd.read_csv(gt_path)
        df.columns = df.columns.str.strip()

        return df

    def get_ground_truth(self, image_id):
        """Get ground truth for a specific image."""

        if self.ground_truth.empty:
            return None

        result = self.ground_truth[
            self.ground_truth["imgid"] == image_id
        ]

        if result.empty:
            return None

        return result.iloc[0]

    # ----------------------------------------------------------
    # Onboard pose (IMU)
    # ----------------------------------------------------------

    @property
    def pose(self):
        """Onboard pose DataFrame, loaded lazily."""

        if self._pose is None:
            self._pose = self._load_pose()

        return self._pose

    def _load_pose(self):
        """Load onboard pose from OnboardPose.csv."""

        pose_path = self.log_dir / "OnboardPose.csv"

        if not pose_path.exists():
            logger.warning("No pose file found: %s", pose_path)
            return pd.DataFrame()

        df = pd.read_csv(pose_path)
        df.columns = df.columns.str.strip()

        return df

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):
        """Print a summary of the dataset."""

        print("=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Root:           {self.root}")
        print(f"Images:         {self.num_images}")

        if self.num_images > 0:
            print(
                f"ID range:       "
                f"{self.image_ids[0]} to {self.image_ids[-1]}"
            )

        print(f"GPS records:    {len(self.gps)}")
        print(f"Ground truth:   {len(self.ground_truth)}")
        print(f"Pose records:   {len(self.pose)}")

    # ----------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------

    @classmethod
    def from_extracted_frames(
        cls,
        frames_dir,
        image_ext=".jpg",
    ):
        """
        Create a dataset from a directory of pre-extracted
        frames (e.g., output of video_ingest).

        No GPS/log data is expected in this mode.
        """

        ds = cls.__new__(cls)
        ds.root = Path(frames_dir)
        ds.image_dir = ds.root
        ds.log_dir = ds.root  # no logs expected
        ds.image_ext = image_ext

        ds._gps = pd.DataFrame()
        ds._pose = pd.DataFrame()
        ds._ground_truth = pd.DataFrame()
        ds._image_index = None

        return ds
