import numpy as np
from typing import Dict, Tuple, Optional
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId


def compute_target_curvatures(
    factory: SurfaceFactory,
    surface_id: SurfaceId,
    target_interior_curvature: Optional[float] = None,
    target_boundary_geodesic_curvature: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Computes a topologically compliant target curvature vector K* satisfying the
    discrete Gauss-Bonnet theorem:
        sum_{v in V_i} (2*pi - K_v) + sum_{v in V_b} (pi - K_v) = 2*pi * chi(Sigma)

    In discrete_conformal_map.py, the linear term is sum_v (C_v - K_v) * u_v,
    where C_v = 2*pi for interior vertices and C_v = pi for boundary vertices.
    This function computes K_v such that sum_v K_v = 2*pi * chi(Sigma).

    Args:
        factory: SurfaceFactory instance managing mesh topology and properties.
        surface_id: SurfaceId for the mesh.
        target_interior_curvature: Total target Gaussian curvature for interior vertices.
        target_boundary_geodesic_curvature: Target geodesic curvature per boundary vertex.

    Returns:
        A tuple (target_curvatures, metadata_dict):
            target_curvatures: 1D NumPy array of size N_vertices with compliant K_i values.
            metadata_dict: Topology and Gauss-Bonnet validation statistics.
    """
    viewer = factory.create_viewer(surface_id)

    num_vertices = len(viewer.vertices)
    num_faces = len(viewer.faces)

    processed_edges = set()
    num_edges = 0
    for he in viewer.half_edges:
        if he.twin:
            edge_key = frozenset({he.id, he.twin.id})
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
        num_edges += 1

    chi = num_vertices - num_edges + num_faces
    total_required_curvature = 2.0 * np.pi * chi

    interior_indices = []
    boundary_indices = []

    for i, v in enumerate(viewer.vertices):
        is_boundary = any(he.twin is None for he in v.half_edges)
        idx = getattr(v, "dcm_index", None)
        if idx is None:
            idx = i

        if is_boundary:
            boundary_indices.append(idx)
        else:
            interior_indices.append(idx)

    num_interior = len(interior_indices)
    num_boundary = len(boundary_indices)

    target_K = np.zeros(num_vertices, dtype=np.float64)

    if target_interior_curvature is not None:
        if num_interior > 0:
            target_K[interior_indices] = target_interior_curvature / num_interior
        remaining_curvature = total_required_curvature - np.sum(target_K[interior_indices])
        if num_boundary > 0:
            target_K[boundary_indices] = remaining_curvature / num_boundary
    elif target_boundary_geodesic_curvature is not None:
        if num_boundary > 0:
            target_K[boundary_indices] = target_boundary_geodesic_curvature
        remaining_curvature = total_required_curvature - np.sum(target_K[boundary_indices])
        if num_interior > 0:
            target_K[interior_indices] = remaining_curvature / num_interior
    else:
        # Default Gauss-Bonnet smooth distribution across boundary or interior
        if num_boundary > 0:
            target_K[boundary_indices] = total_required_curvature / num_boundary
        elif num_interior > 0:
            target_K[interior_indices] = total_required_curvature / num_interior

    actual_sum = float(np.sum(target_K))
    compliant = bool(np.isclose(actual_sum, total_required_curvature, atol=1e-8))

    metadata = {
        "euler_characteristic": chi,
        "num_vertices": num_vertices,
        "num_interior_vertices": num_interior,
        "num_boundary_vertices": num_boundary,
        "total_required_gauss_bonnet_curvature": total_required_curvature,
        "actual_allocated_target_curvature_sum": actual_sum,
        "gauss_bonnet_compliant": compliant,
    }

    return target_K, metadata
