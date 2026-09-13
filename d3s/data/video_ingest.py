"""
Video ingestion — extract frames from drone video files.

Handles:
  - Frame extraction at configurable FPS
  - Resolution capping
  - Timestamp synchronization
  - Output as numbered JPEG sequence
"""

import logging
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path,
    output_dir,
    target_fps=None,
    max_resolution=None,
    frame_format="jpg",
    jpeg_quality=95,
):
    """
    Extract frames from a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    output_dir : str or Path
        Directory to write extracted frames.
    target_fps : float, optional
        Target frames per second. If None, extracts all frames.
    max_resolution : int, optional
        Maximum resolution (longest edge). Frames larger than
        this are downscaled proportionally.
    frame_format : str
        Output image format ("jpg" or "png").
    jpeg_quality : int
        JPEG quality (1-100). Only used if format is "jpg".

    Returns
    -------
    dict
        Extraction metadata:
        - frame_count: number of frames extracted
        - original_fps: source video FPS
        - timestamps: list of frame timestamps (seconds)
        - frame_paths: list of output file paths
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # Video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video: %s — %dx%d @ %.1f FPS, %d frames",
        video_path.name,
        width,
        height,
        original_fps,
        total_frames,
    )

    # Determine frame sampling interval
    if target_fps is not None and target_fps < original_fps:
        frame_interval = original_fps / target_fps
    else:
        frame_interval = 1.0

    # Determine resize dimensions
    scale = _compute_scale(width, height, max_resolution)

    # Encode parameters
    if frame_format == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
        ext = ".png"
        params = []

    # Extract frames
    frame_paths = []
    timestamps = []

    frame_idx = 0
    next_sample = 0.0
    extracted = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx >= next_sample:

            # Resize if needed
            if scale is not None:
                new_w = int(width * scale)
                new_h = int(height * scale)

                frame = cv2.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA,
                )

            # Save
            filename = f"frame_{extracted:06d}{ext}"
            out_path = output_dir / filename

            cv2.imwrite(str(out_path), frame, params)

            frame_paths.append(out_path)

            timestamp = frame_idx / original_fps
            timestamps.append(timestamp)

            extracted += 1
            next_sample += frame_interval

        frame_idx += 1

        # Progress reporting every 500 frames
        if frame_idx % 500 == 0:
            progress = frame_idx / total_frames * 100
            logger.info(
                "Extracting: %d/%d (%.0f%%)",
                frame_idx,
                total_frames,
                progress,
            )

    cap.release()

    logger.info(
        "Extracted %d frames from %d total (%.1fx reduction)",
        extracted,
        total_frames,
        total_frames / max(1, extracted),
    )

    return {
        "frame_count": extracted,
        "original_fps": original_fps,
        "total_source_frames": total_frames,
        "timestamps": timestamps,
        "frame_paths": frame_paths,
    }


def _compute_scale(
    width,
    height,
    max_resolution,
):
    """
    Compute the scale factor to fit within max_resolution.
    Returns None if no resizing is needed.
    """

    if max_resolution is None:
        return None

    longest = max(width, height)

    if longest <= max_resolution:
        return None

    return max_resolution / longest
