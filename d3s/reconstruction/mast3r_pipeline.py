"""
Core MASt3R-SfM reconstruction pipeline.

Replaces:
  - hybrid_dust3r_pilot.py (DUSt3R inference + alignment)
  - align_trajectories.py (Umeyama transform)
  - test_dense_alignment.py (DUSt3R->COLMAP alignment)

MASt3R-SfM produces metric-scale reconstructions directly,
eliminating the need for external COLMAP and Umeyama alignment.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from d3s.config import ReconstructionConfig, resolve_device

logger = logging.getLogger(__name__)

# Path to the project root (d3s/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_mast3r_path():
    """
    Add MASt3R and DUSt3R to Python's import path.

    MASt3R doesn't use a standard pip-installable layout,
    so we add it to sys.path at runtime. Searches in:
      1. third_party/mast3r  (fresh clone)
      2. MAST3R_ROOT environment variable
    """

    import os

    # Check if already importable
    try:
        import mast3r  # noqa: F401
        return
    except ImportError:
        pass

    # Try standard location
    candidates = [
        _PROJECT_ROOT / "third_party" / "mast3r",
    ]

    env_root = os.environ.get("MAST3R_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))

    for root in candidates:
        if (root / "mast3r").is_dir():
            mast3r_path = str(root)
            dust3r_path = str(root / "dust3r")

            if mast3r_path not in sys.path:
                sys.path.insert(0, mast3r_path)
                logger.debug(
                    "Added to sys.path: %s", mast3r_path
                )

            if dust3r_path not in sys.path:
                sys.path.insert(0, dust3r_path)
                logger.debug(
                    "Added to sys.path: %s", dust3r_path
                )

            return

    raise ImportError(
        "MASt3R not found. Expected at "
        f"{_PROJECT_ROOT / 'third_party' / 'mast3r'}\n"
        "Clone it with: git clone --recursive "
        "https://github.com/naver/mast3r "
        "third_party/mast3r"
    )


class MASt3RPipeline:
    """
    End-to-end 3D reconstruction using MASt3R-SfM.

    Usage::

        pipeline = MASt3RPipeline(config)
        pipeline.load_model()
        result = pipeline.reconstruct(image_paths)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : ReconstructionConfig, optional
            Reconstruction parameters. Uses defaults if None.
        """

        if config is None:
            config = ReconstructionConfig()

        self.config = config
        self.device_name = resolve_device(config)
        self.model = None

        logger.info(
            "MASt3R pipeline initialized (device=%s)",
            self.device_name,
        )

    def load_model(self, model_name=None):
        """
        Load the MASt3R model.

        Parameters
        ----------
        model_name : str, optional
            Model identifier or path. If None, uses the
            config default.
        """

        import os
        import torch

        # Fix conda SSL_CERT_FILE issue before network calls
        _ssl = os.environ.get("SSL_CERT_FILE", "")
        if _ssl and not Path(_ssl).exists():
            del os.environ["SSL_CERT_FILE"]

        # Lazy imports — only needed when actually running
        _setup_mast3r_path()
        from mast3r.model import AsymmetricMASt3R

        if model_name is None:
            model_name = (
                "naver/MASt3R_ViTLarge_BaseDecoder_512"
                "_catmlpdpt_metric"
            )

        logger.info("Loading MASt3R model: %s", model_name)

        t0 = time.time()

        try:
            self.model = (
                AsymmetricMASt3R.from_pretrained(model_name)
            )
        except (OSError, FileNotFoundError, Exception) as e:
            # If online fetch fails, try offline (cached model)
            logger.warning(
                "Online model fetch failed (%s), "
                "trying cached model...",
                type(e).__name__,
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = (
                AsymmetricMASt3R.from_pretrained(model_name)
            )

        self.device = torch.device(self.device_name)
        self.model = self.model.to(self.device)

        logger.info(
            "Model loaded in %.1fs on %s",
            time.time() - t0,
            self.device_name,
        )

    def reconstruct(
        self,
        image_paths,
        scene_graph=None,
    ):
        """
        Run full 3D reconstruction on a set of images.

        Parameters
        ----------
        image_paths : list of str or Path
            Ordered list of images to reconstruct.
        scene_graph : str, optional
            Scene graph type. If None, uses config default.

        Returns
        -------
        ReconstructionResult
            Object containing point clouds, camera poses,
            confidence maps, etc.
        """

        import torch

        _setup_mast3r_path()
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import (
            global_aligner,
            GlobalAlignerMode,
        )

        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        if scene_graph is None:
            scene_graph = self.config.scene_graph

        image_paths = [str(p) for p in image_paths]

        n_images = len(image_paths)

        logger.info(
            "Reconstructing %d images (scene_graph=%s)",
            n_images,
            scene_graph,
        )

        # --- Load images ---
        t0 = time.time()

        imgs = load_images(
            image_paths,
            size=self.config.image_size,
            verbose=True,
        )

        logger.info(
            "Images loaded in %.1fs", time.time() - t0
        )

        # --- Build pairs ---
        t0 = time.time()

        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True,
        )

        logger.info(
            "Built %d pairs in %.1fs",
            len(pairs),
            time.time() - t0,
        )

        # --- Run inference ---
        t0 = time.time()

        output = inference(
            pairs,
            self.model,
            self.device,
            batch_size=1,
            verbose=True,
        )

        logger.info(
            "Inference completed in %.1fs",
            time.time() - t0,
        )

        # --- Global alignment ---
        t0 = time.time()

        mode = GlobalAlignerMode.PointCloudOptimizer

        scene = global_aligner(
            output,
            device=self.device,
            mode=mode,
        )

        loss = scene.compute_global_alignment(
            init=self.config.alignment_init,
            niter=self.config.alignment_iterations,
            schedule=self.config.alignment_schedule,
            lr=self.config.alignment_lr,
        )

        logger.info(
            "Global alignment completed in %.1fs "
            "(final loss=%.6f)",
            time.time() - t0,
            float(loss),
        )

        # --- Extract results ---
        result = self._extract_results(
            scene, image_paths
        )

        return result

    def _extract_results(self, scene, image_paths):
        """
        Extract camera poses, point clouds, and confidence
        from the aligned scene.
        """

        import torch

        # Camera-to-world poses [N, 4, 4]
        im_poses = (
            scene.get_im_poses()
            .detach()
            .cpu()
            .numpy()
        )

        # Camera centers [N, 3]
        centers = im_poses[:, :3, 3]

        # Focal lengths
        try:
            focals = (
                scene.get_focals()
                .detach()
                .cpu()
                .numpy()
            )
        except Exception:
            focals = None

        # Per-image point maps and confidence
        pts3d_list = []
        masks_list = []
        conf_list = []

        for i in range(len(image_paths)):
            try:
                pts = (
                    scene.get_pts3d()[i]
                    .detach()
                    .cpu()
                    .numpy()
                )
                mask = (
                    scene.get_masks()[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                pts3d_list.append(pts)
                masks_list.append(mask)

                try:
                    c = (
                        scene.im_conf[i]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                except Exception:
                    c = np.ones(
                        mask.shape, dtype=np.float32
                    )

                conf_list.append(c)

            except Exception as e:
                logger.warning(
                    "Failed to extract data for image %d: %s",
                    i,
                    e,
                )

        # Build merged point cloud
        all_points, all_conf = (
            self._merge_point_clouds(
                pts3d_list, masks_list, conf_list
            )
        )

        return ReconstructionResult(
            image_paths=image_paths,
            im_poses=im_poses,
            camera_centers=centers,
            focals=focals,
            pts3d=pts3d_list,
            masks=masks_list,
            confidence=conf_list,
            merged_points=all_points,
            merged_confidence=all_conf,
        )

    def _merge_point_clouds(
        self, pts3d_list, masks_list, conf_list
    ):
        """
        Merge per-image point maps into a single cloud,
        filtering by confidence.
        """

        all_points = []
        all_conf = []

        for pts, mask, conf in zip(
            pts3d_list, masks_list, conf_list
        ):

            pts = np.asarray(pts, dtype=np.float64)
            mask = np.asarray(mask, dtype=bool)
            conf = np.asarray(conf, dtype=np.float64)

            # Flatten H x W x 3
            pts_flat = pts.reshape(-1, 3)
            mask_flat = mask.reshape(-1)
            conf_flat = conf.reshape(-1)

            valid = (
                mask_flat
                & np.isfinite(pts_flat).all(axis=1)
                & np.isfinite(conf_flat)
            )

            pts_valid = pts_flat[valid]
            conf_valid = conf_flat[valid]

            # Remove low-confidence points
            if len(pts_valid) > 0:
                threshold = np.percentile(
                    conf_valid,
                    self.config.confidence_threshold_percentile,
                )
                keep = conf_valid >= threshold
                pts_valid = pts_valid[keep]
                conf_valid = conf_valid[keep]

            all_points.append(pts_valid)
            all_conf.append(conf_valid)

        if all_points:
            merged_pts = np.concatenate(all_points, axis=0)
            merged_conf = np.concatenate(all_conf, axis=0)
        else:
            merged_pts = np.empty((0, 3))
            merged_conf = np.empty((0,))

        # Downsample if too many points
        if len(merged_pts) > self.config.max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(
                len(merged_pts),
                size=self.config.max_points,
                replace=False,
            )
            merged_pts = merged_pts[idx]
            merged_conf = merged_conf[idx]

        logger.info(
            "Merged point cloud: %d points", len(merged_pts)
        )

        return merged_pts, merged_conf


class ReconstructionResult:
    """
    Container for reconstruction outputs.

    Attributes
    ----------
    image_paths : list of str
        Input image file paths.
    im_poses : ndarray [N, 4, 4]
        Camera-to-world transformation matrices.
    camera_centers : ndarray [N, 3]
        Camera center positions in the reconstruction frame.
    focals : ndarray [N] or None
        Estimated focal lengths.
    pts3d : list of ndarray
        Per-image 3D point maps (H x W x 3).
    masks : list of ndarray
        Per-image validity masks (H x W).
    confidence : list of ndarray
        Per-image confidence maps (H x W).
    merged_points : ndarray [M, 3]
        Merged 3D point cloud.
    merged_confidence : ndarray [M]
        Confidence values for merged points.
    """

    def __init__(
        self,
        image_paths,
        im_poses,
        camera_centers,
        focals,
        pts3d,
        masks,
        confidence,
        merged_points,
        merged_confidence,
    ):
        self.image_paths = image_paths
        self.im_poses = im_poses
        self.camera_centers = camera_centers
        self.focals = focals
        self.pts3d = pts3d
        self.masks = masks
        self.confidence = confidence
        self.merged_points = merged_points
        self.merged_confidence = merged_confidence

    @property
    def num_images(self):
        return len(self.image_paths)

    @property
    def num_points(self):
        return len(self.merged_points)

    def trajectory_length(self):
        """Total camera trajectory length."""

        dists = np.linalg.norm(
            self.camera_centers[1:]
            - self.camera_centers[:-1],
            axis=1,
        )
        return float(dists.sum())

    def bounding_box(self):
        """Point cloud axis-aligned bounding box."""

        if len(self.merged_points) == 0:
            return None

        return {
            "min": self.merged_points.min(axis=0),
            "max": self.merged_points.max(axis=0),
            "size": (
                self.merged_points.max(axis=0)
                - self.merged_points.min(axis=0)
            ),
        }

    def save(self, output_dir):
        """
        Save reconstruction data to a directory.

        Saves:
          - scene.npz: all reconstruction data
          - camera_trajectory.csv: camera centers
        """

        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scene data
        np.savez_compressed(
            output_dir / "scene.npz",
            image_paths=np.array(
                self.image_paths, dtype=object
            ),
            im_poses=self.im_poses,
            camera_centers=self.camera_centers,
            focals=(
                self.focals
                if self.focals is not None
                else np.array([])
            ),
            merged_points=self.merged_points,
            merged_confidence=self.merged_confidence,
        )

        # Save camera trajectory
        traj_path = output_dir / "camera_trajectory.csv"

        with open(traj_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image", "cx", "cy", "cz"]
            )

            for path, center in zip(
                self.image_paths,
                self.camera_centers,
            ):
                name = Path(path).name
                writer.writerow(
                    [
                        name,
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                    ]
                )

        logger.info("Saved reconstruction to %s", output_dir)

        return {
            "scene": output_dir / "scene.npz",
            "trajectory": traj_path,
        }
