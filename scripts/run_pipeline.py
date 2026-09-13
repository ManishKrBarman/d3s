"""
D3S Pipeline — End-to-end CLI entry point.

Runs the full pipeline:
  1. Video ingestion (if input is a video file)
  2. Quality filtering
  3. Keyframe selection
  4. MASt3R-SfM reconstruction
  5. Georeferencing (if GPS data available)
  6. Mesh generation
  7. Export

Usage:
    python scripts/run_pipeline.py --config config/default.yaml --input data/AGZ_subset
    python scripts/run_pipeline.py --config config/default.yaml --input path/to/video.mp4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix conda SSL_CERT_FILE pointing to non-existent file
# (common issue with conda environments on Windows)
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not Path(_ssl_cert).exists():
    del os.environ["SSL_CERT_FILE"]


def setup_logging(verbose=False):
    """Configure logging."""

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="D3S: Drone Video to 3D Model Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input path: video file (.mp4/.mov) or "
            "image directory"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Compute device (overrides config)",
    )

    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Skip MASt3R reconstruction (for testing)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("d3s.pipeline")

    t_start = time.time()

    # ----------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------
    from d3s.config import load_config

    overrides = {}

    if args.output:
        overrides["paths"] = {"output_root": args.output}

    if args.device:
        overrides["reconstruction"] = {"device": args.device}

    config = load_config(args.config, overrides)

    output_dir = Path(config.paths.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("D3S PIPELINE")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", output_dir)

    # ----------------------------------------------------------
    # Step 1: Determine input type and load data
    # ----------------------------------------------------------
    input_path = Path(args.input)

    video_extensions = {
        ".mp4", ".mov", ".avi", ".mkv", ".webm"
    }

    if (
        input_path.is_file()
        and input_path.suffix.lower() in video_extensions
    ):
        # Video input — extract frames first
        logger.info("Input is a video file, extracting frames...")

        from d3s.data.video_ingest import extract_frames

        frames_dir = output_dir / "frames"

        result = extract_frames(
            video_path=input_path,
            output_dir=frames_dir,
            target_fps=config.video.target_fps,
            max_resolution=config.video.max_resolution,
            frame_format=config.video.frame_format,
            jpeg_quality=config.video.jpeg_quality,
        )

        logger.info(
            "Extracted %d frames", result["frame_count"]
        )

        from d3s.data.dataset import Dataset

        dataset = Dataset.from_extracted_frames(frames_dir)

    elif input_path.is_dir():
        # Directory input — load as dataset
        from d3s.data.dataset import Dataset

        dataset = Dataset(input_path)

    else:
        logger.error(
            "Input not found or unrecognized format: %s",
            input_path,
        )
        sys.exit(1)

    dataset.summary()

    if dataset.num_images == 0:
        logger.error("No images found!")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 2: Quality filtering
    # ----------------------------------------------------------
    logger.info("\n--- QUALITY FILTERING ---")

    from d3s.preprocessing.quality_filter import (
        filter_by_quality,
    )

    quality_result = filter_by_quality(
        image_paths=dataset.image_paths,
        min_sharpness=config.quality.min_sharpness,
        min_sharpness_percentile=(
            config.keyframes.min_sharpness_percentile
        ),
        min_brightness=config.quality.min_brightness,
        max_brightness=config.quality.max_brightness,
        min_contrast=config.quality.min_contrast,
    )

    passed_paths = [p for p, _ in quality_result["passed"]]

    logger.info(
        "Quality: %d/%d passed",
        len(passed_paths),
        dataset.num_images,
    )

    # ----------------------------------------------------------
    # Step 3: Keyframe selection
    # ----------------------------------------------------------
    logger.info("\n--- KEYFRAME SELECTION ---")

    from d3s.preprocessing.keyframe_selector import (
        select_keyframes,
    )

    quality_map = {
        p: m for p, m in quality_result["passed"]
    }

    keyframe_result = select_keyframes(
        image_paths=passed_paths,
        config=config.keyframes,
        quality_metrics=quality_map,
    )

    selected_paths = keyframe_result["selected"]

    logger.info(
        "Keyframes: %d selected",
        len(selected_paths),
    )

    for key, val in keyframe_result["stats"].items():
        logger.info("  %s: %s", key, val)

    # ----------------------------------------------------------
    # Step 4: MASt3R-SfM reconstruction
    # ----------------------------------------------------------
    if not args.skip_reconstruction:
        logger.info("\n--- 3D RECONSTRUCTION ---")

        from d3s.reconstruction.mast3r_pipeline import (
            MASt3RPipeline,
        )

        pipeline = MASt3RPipeline(config.reconstruction)
        pipeline.load_model(config.paths.mast3r_model)

        reconstruction = pipeline.reconstruct(selected_paths)

        logger.info(
            "Reconstruction: %d points, %d cameras",
            reconstruction.num_points,
            reconstruction.num_images,
        )

        logger.info(
            "Trajectory length: %.4f",
            reconstruction.trajectory_length(),
        )

        bbox = reconstruction.bounding_box()

        if bbox:
            logger.info("Bounding box: %s", bbox["size"])

        # Save reconstruction data
        recon_dir = output_dir / "reconstruction"
        reconstruction.save(recon_dir)

        # -------------------------------------------------------
        # Step 5: Georeferencing
        # -------------------------------------------------------
        if (
            config.georef.enabled
            and not dataset.gps.empty
        ):
            logger.info("\n--- GEOREFERENCING ---")

            from d3s.reconstruction.georef import (
                georeference,
            )

            georef_result = georeference(
                reconstruction,
                dataset.gps,
                earth_radius=config.georef.earth_radius,
            )

            from d3s.export.geotiff import (
                export_georef_metadata,
            )

            export_georef_metadata(
                georef_result,
                output_dir / "georef",
            )

        # -------------------------------------------------------
        # Step 6: Export point cloud
        # -------------------------------------------------------
        logger.info("\n--- EXPORT ---")

        from d3s.export.ply_export import export_point_cloud

        for fmt in config.export.formats:
            export_path = (
                output_dir / f"reconstruction.{fmt}"
            )

            export_point_cloud(
                points=reconstruction.merged_points,
                output_path=export_path,
                confidence=reconstruction.merged_confidence,
                format=fmt,
            )

        # -------------------------------------------------------
        # Step 7: Mesh generation
        # -------------------------------------------------------
        if config.mesh.enabled:
            logger.info("\n--- MESH GENERATION ---")

            from d3s.reconstruction.mesh import (
                generate_mesh,
            )

            try:
                mesh = generate_mesh(
                    points=reconstruction.merged_points,
                    config=config.mesh,
                    output_path=output_dir / "mesh.ply",
                )
            except Exception as e:
                logger.warning(
                    "Mesh generation failed: %s", e
                )

    else:
        logger.info(
            "\nSkipping reconstruction (--skip-reconstruction)"
        )

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    elapsed = time.time() - t_start

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time: %.1f seconds", elapsed)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
