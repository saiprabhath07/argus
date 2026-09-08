# ARGUS - mesh helpers using Open3D
# Add open3d to requirements to enable this module

import os
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False


def load_ply_to_numpy(ply_path):
    """Load a PLY point cloud into numpy arrays (xyz, rgb).
    Returns (xyz (N,3) float64, rgb (N,3) uint8 or None)
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is not installed")

    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = None
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors, dtype=np.float64) * 255.0).astype(np.uint8)
    return xyz, rgb


def generate_mesh_from_points(xyz, rgb=None, method='poisson', poisson_depth=9, poisson_scale=1.1, poisson_linear_fit=False, bpa_radii=None, simplify_triangles=None):
    """Generate a triangle mesh from point cloud using Open3D.

    method: 'poisson' or 'bpa'
    poisson_depth: int (8-12 typical)
    bpa_radii: list of radii (meters in point cloud units) for Ball Pivoting
    simplify_triangles: target triangle count (int) or None
    """
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is not installed")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector((rgb / 255.0).astype(np.float64))

    # Estimate normals
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    # heuristic radius for normal estimation
    radius = max(0.01 * diag, diag * 0.02)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    if method == 'poisson':
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth, scale=poisson_scale, linear_fit=poisson_linear_fit)
        densities = np.asarray(densities)
        # prune low-density vertices (remove the bottom 1% by density)
        if densities.size > 0:
            thresh = np.quantile(densities, 0.01)
            mask = densities > thresh
            mesh = mesh.remove_vertices_by_mask(~mask)
        # crop to original bbox to remove some floating components
        mesh = mesh.crop(bbox)
    elif method == 'bpa':
        if bpa_radii is None:
            # estimate radii based on bbox diagonal
            r = max(0.01 * diag, diag * 0.02)
            bpa_radii = [r, r*2, r*4]
        radii = o3d.utility.DoubleVector(bpa_radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        mesh = mesh.crop(bbox)
    else:
        raise ValueError("method must be 'poisson' or 'bpa'")

    # Simplify if requested
    if simplify_triangles is not None and len(mesh.triangles) > simplify_triangles:
        mesh = mesh.simplify_quadric_decimation(simplify_triangles)

    # If vertex colors absent but input had colors, sample from pcd
    if rgb is not None and not mesh.has_vertex_colors():
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = []
        for v in mesh.vertices:
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            mesh_colors.append(pcd.colors[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    return mesh


def save_mesh(mesh, out_base_path):
    """Save mesh as OBJ and PLY (ASCII). Returns (obj_path, ply_path)."""
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is not installed")
    base, _ = os.path.splitext(out_base_path)
    obj_path = base + ".obj"
    ply_path = base + ".ply"
    # Write OBJ and PLY (ASCII) to preserve portability
    o3d.io.write_triangle_mesh(obj_path, mesh, write_ascii=True)
    o3d.io.write_triangle_mesh(ply_path, mesh, write_ascii=True)
    return obj_path, ply_path
