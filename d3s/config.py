"""
Configuration loading and validation for the D3S pipeline.

Loads YAML configuration files and provides typed access
to all pipeline parameters. Supports overriding via CLI
arguments or environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ==============================================================
# Configuration dataclasses
# ==============================================================


@dataclass
class PathsConfig:
    """File system paths."""

    data_root: str = "data"
    output_root: str = "output"
    mast3r_model: str = (
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )


@dataclass
class VideoConfig:
    """Video ingestion parameters."""

    target_fps: Optional[float] = None
    max_resolution: Optional[int] = 1920
    frame_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass
class KeyframeConfig:
    """Keyframe selection parameters."""

    strategy: str = "adaptive"

    # Adaptive strategy
    min_sharpness_percentile: int = 20
    min_frame_gap: int = 3
    max_frame_gap: int = 10
    n_features: int = 3000
    match_distance: float = 50
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.99
    high_redundancy: float = 0.85
    low_redundancy: float = 0.70

    # Uniform strategy
    uniform_interval: int = 5


@dataclass
class QualityConfig:
    """Image quality filtering parameters."""

    min_sharpness: Optional[float] = None
    min_brightness: float = 30
    max_brightness: float = 240
    min_contrast: float = 15


@dataclass
class ReconstructionConfig:
    """MASt3R reconstruction parameters."""

    device: str = "auto"
    min_vram_gb: float = 3.0
    image_size: int = 512
    scene_graph: str = "swin-5"
    alignment_iterations: int = 300
    alignment_lr: float = 0.01
    alignment_init: str = "mst"
    alignment_schedule: str = "cosine"
    confidence_threshold_percentile: int = 20
    max_points: int = 1_000_000


@dataclass
class GeorefConfig:
    """Georeferencing parameters."""

    enabled: bool = True
    output_crs: str = "utm"
    earth_radius: float = 6371000.0


@dataclass
class MeshConfig:
    """Mesh generation parameters."""

    enabled: bool = True
    method: str = "poisson"
    poisson_depth: int = 9
    normal_neighbors: int = 30


@dataclass
class ExportConfig:
    """Output export parameters."""

    formats: list = field(default_factory=lambda: ["ply"])
    vertex_colors: bool = True
    vertex_normals: bool = True
    save_trajectory: bool = True
    save_statistics: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    reconstruction: ReconstructionConfig = field(
        default_factory=ReconstructionConfig
    )
    georef: GeorefConfig = field(default_factory=GeorefConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


# ==============================================================
# Loading
# ==============================================================


def _merge_dict_into_dataclass(dc, data: dict):
    """
    Recursively update a dataclass instance from a dictionary.
    Only updates fields that exist in the dataclass.
    """

    if data is None:
        return dc

    for key, value in data.items():

        if not hasattr(dc, key):
            continue

        current = getattr(dc, key)

        # If the current attribute is a dataclass, recurse
        if hasattr(current, "__dataclass_fields__") and isinstance(
            value, dict
        ):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)

    return dc


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """
    Load pipeline configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file. If None, uses defaults.
    overrides : dict, optional
        Dictionary of overrides to apply on top of the file config.

    Returns
    -------
    PipelineConfig
        Fully resolved configuration object.
    """

    config = PipelineConfig()

    # Load from YAML file
    if config_path is not None:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is not None:
            _merge_dict_into_dataclass(config, data)

    # Apply overrides
    if overrides is not None:
        _merge_dict_into_dataclass(config, overrides)

    # Resolve environment variable overrides
    env_model = os.environ.get("D3S_MAST3R_MODEL")

    if env_model:
        config.paths.mast3r_model = env_model

    env_device = os.environ.get("D3S_DEVICE")

    if env_device:
        config.reconstruction.device = env_device

    return config


def resolve_device(config: ReconstructionConfig) -> str:
    """
    Determine the compute device to use.

    Returns "cuda" or "cpu" based on config and hardware.
    """

    if config.device == "cpu":
        return "cpu"

    if config.device == "cuda":
        return "cuda"

    # Auto-detect
    try:
        import torch

        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = vram_bytes / (1024 ** 3)

            if vram_gb >= config.min_vram_gb:
                return "cuda"

            print(
                f"[D3S] GPU has {vram_gb:.1f}GB VRAM, "
                f"need {config.min_vram_gb}GB. "
                f"Falling back to CPU."
            )

        return "cpu"

    except ImportError:
        return "cpu"
