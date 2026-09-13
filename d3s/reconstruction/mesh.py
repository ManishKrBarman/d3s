"""
Mesh generation from 3D point clouds.

Converts the MASt3R dense point cloud into a surface mesh
using Poisson or Ball Pivoting reconstruction.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from d3s.config import MeshConfig

logger = logging.getLogger(__name__)


def generate_mesh(
    points,
    colors=None,
    config=None,
    output_path=None,
):
    """
    Generate a surface mesh from a 3D point cloud.

    Parameters
    ----------
    points : ndarray [N, 3]
        3D point positions.
    colors : ndarray [N, 3], optional
        Per-point RGB colors (0-1 range).
    config : MeshConfig, optional
        Mesh generation parameters.
    output_path : str or Path, optional
        If provided, saves the mesh to this file.

    Returns
    -------
    open3d.geometry.TriangleMesh
        The generated mesh.
    """

    import open3d as o3d

    if config is None:
        config = MeshConfig()

    logger.info(
        "Generating mesh from %d points (method=%s)",
        len(points),
        config.method,
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals
    logger.info(
        "Estimating normals (k=%d)...",
        config.normal_neighbors,
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(
            knn=config.normal_neighbors
        )
    )

    pcd.orient_normals_consistent_tangent_plane(
        k=config.normal_neighbors
    )

    # Generate mesh
    if config.method == "poisson":
        mesh = _poisson_reconstruction(pcd, config)
    elif config.method == "ball_pivoting":
        mesh = _ball_pivoting(pcd)
    else:
        raise ValueError(
            f"Unknown mesh method: {config.method}"
        )

    logger.info(
        "Mesh generated: %d vertices, %d triangles",
        len(mesh.vertices),
        len(mesh.triangles),
    )

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_triangle_mesh(
            str(output_path), mesh
        )

        logger.info("Mesh saved to %s", output_path)

    return mesh


def _poisson_reconstruction(pcd, config):
    """Poisson surface reconstruction."""

    import open3d as o3d

    logger.info(
        "Running Poisson reconstruction (depth=%d)...",
        config.poisson_depth,
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            pcd,
            depth=config.poisson_depth,
        )
    )

    # Remove low-density vertices (noise at edges)
    densities = np.asarray(densities)

    density_threshold = np.percentile(densities, 5)

    vertices_to_remove = densities < density_threshold

    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def _ball_pivoting(pcd):
    """Ball-pivoting surface reconstruction."""

    import open3d as o3d

    # Estimate ball radii from point spacing
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radii = [
        avg_dist * 1.0,
        avg_dist * 2.0,
        avg_dist * 4.0,
    ]

    logger.info(
        "Running Ball Pivoting (radii=%.4f, %.4f, %.4f)...",
        *radii,
    )

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
    )

    return mesh


def save_point_cloud(
    points,
    output_path,
    colors=None,
    normals=None,
):
    """
    Save a point cloud to a PLY file.

    Parameters
    ----------
    points : ndarray [N, 3]
    output_path : str or Path
    colors : ndarray [N, 3], optional (0-1 range)
    normals : ndarray [N, 3], optional
    """

    import open3d as o3d

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(str(output_path), pcd)

    logger.info(
        "Point cloud saved: %s (%d points)",
        output_path,
        len(points),
    )
