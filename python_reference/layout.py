# mazedemo/layout.py
"""
Canonical Layout Generator and Polygon Normalizer for Module 3.

Handles Minkowski light-cone projection to complex Poincaré disk coordinates (z in C),
Canonical Möbius Centering for Hyperbolic geometry, and Holonomy Generator Anchoring
for Euclidean geometry.
"""

import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from mazedemo.ids import FaceId, VertexId, SurfaceId
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.fundamental_domain import compute_fundamental_domain
from mazedemo.hyperbolic import HyperbolicGeometry
from mazedemo.euclidean import EuclideanGeometry


def to_cartesian(z: Any) -> np.ndarray:
    """
    Converts complex Poincaré coordinates or dicts of complex coordinates
    to float64 N x 2 Cartesian coordinates [x, y].
    """
    if isinstance(z, dict):
        return {v_id: to_cartesian(val) for v_id, val in z.items()}
    if isinstance(z, (complex, np.complexfloating)):
        return np.array([float(z.real), float(z.imag)], dtype=np.float64)
    arr = np.asarray(z)
    if np.iscomplexobj(arr):
        return np.column_stack([arr.real.astype(np.float64), arr.imag.astype(np.float64)])
    if arr.ndim == 1 and arr.shape[0] >= 2:
        return arr[:2].astype(np.float64)
    return arr.astype(np.float64)


def mobius_transform(z: np.complex128, z_c: np.complex128) -> np.complex128:
    """
    Applies the isometric Möbius centering transformation:
    m_{z_c}(z) = (z - z_c) / (1 - conj(z_c) * z)
    """
    if np.abs(z_c) < 1e-12:
        return z
    num = z - z_c
    den = 1.0 - np.conj(z_c) * z
    if np.abs(den) < 1e-15:
        return z
    return num / den


def generate_canonical_layout(
    factory: SurfaceFactory,
    surface_id: SurfaceId,
    geometry: Any,
) -> Tuple[Dict[FaceId, Dict[VertexId, Any]], Dict[str, Any]]:
    """
    Generates a canonical 2D layout framing a single, cut-open fundamental domain tile.

    For Hyperbolic geometry:
    - Stereographically projects Minkowski hyperboloid coordinates to complex Poincaré coordinates (z in C, complex128).
    - Applies Canonical Möbius Centering to pull the visual spatial centroid z_c to 0.

    For Euclidean geometry:
    - Translates basepoint v0 to origin (0,0).
    - Anchors the layout using primary holonomy generator a1 (aligning g_{a1} with +x axis).

    Returns:
        placed_triangles: Dict[FaceId, Dict[VertexId, complex128 or np.ndarray]]
        metadata: Dict containing domain artifacts, holonomy transforms, etc.
    """
    (
        dual_graph,
        centered_embeddings,
        isometries,
        spanning_tree,
        holonomy_classes,
        raw_placed_triangles,
        holonomy_transforms,
        generators,
    ) = compute_fundamental_domain(factory, surface_id, geometry)

    is_hyperbolic = isinstance(geometry, HyperbolicGeometry)
    placed_triangles: Dict[FaceId, Dict[VertexId, Any]] = {}

    if is_hyperbolic:
        # 1. Project Minkowski hyperboloid points (x, y, z) to complex Poincaré disk z = (x + i y) / (1 + z)
        poincare_placed: Dict[FaceId, Dict[VertexId, np.complex128]] = {}

        for f_id, placement in raw_placed_triangles.items():
            poincare_placed[f_id] = {}
            for v_id, p in placement.items():
                z_val = np.complex128((p[0] + 1j * p[1]) / (1.0 + p[2]))
                poincare_placed[f_id][v_id] = z_val

        # 2. Canonical Möbius Centering via fixed-point iterations
        curr_placed = poincare_placed
        for _ in range(5):
            all_z = [z for placement in curr_placed.values() for z in placement.values()]
            if not all_z:
                break
            z_c = np.complex128(np.mean(all_z))
            if np.abs(z_c) < 1e-10:
                break
            if np.abs(z_c) >= 1.0:
                z_c = z_c / (np.abs(z_c) + 1e-5)

            next_placed: Dict[FaceId, Dict[VertexId, np.complex128]] = {}
            for f_id, placement in curr_placed.items():
                next_placed[f_id] = {}
                for v_id, z_val in placement.items():
                    next_placed[f_id][v_id] = mobius_transform(z_val, z_c)
            curr_placed = next_placed

        placed_triangles = curr_placed

    else:  # Euclidean Geometry
        # Raw placement gives homogeneous coordinates [x, y, 1]
        euc_placed: Dict[FaceId, Dict[VertexId, np.ndarray]] = {}
        for f_id, placement in raw_placed_triangles.items():
            euc_placed[f_id] = {
                v_id: p[:2].astype(np.float64) for v_id, p in placement.items()
            }

        # Find basepoint v0 (origin of root face or first vertex of root face)
        root_face_id = next(iter(euc_placed.keys()))
        root_v_id = next(iter(euc_placed[root_face_id].keys()))
        basepoint_pos = euc_placed[root_face_id][root_v_id].copy()

        # Translate layout so basepoint_pos is at (0, 0)
        translated_placed: Dict[FaceId, Dict[VertexId, np.ndarray]] = {}
        for f_id, placement in euc_placed.items():
            translated_placed[f_id] = {
                v_id: pos - basepoint_pos for v_id, pos in placement.items()
            }

        # Holonomy Generator Anchoring
        # Look for primary generator 'a' transform
        g_a_vec: Optional[np.ndarray] = None
        g_b_vec: Optional[np.ndarray] = None

        for gen_name, T in holonomy_transforms.items():
            if gen_name == "a" and T.shape == (3, 3):
                g_a_vec = T[:2, 2]
            elif gen_name == "b" and T.shape == (3, 3):
                g_b_vec = T[:2, 2]

        angle_rot = 0.0
        if g_a_vec is not None and np.linalg.norm(g_a_vec) > 1e-9:
            angle_rot = -np.arctan2(g_a_vec[1], g_a_vec[0])

        cos_a, sin_a = np.cos(angle_rot), np.sin(angle_rot)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation
        for f_id, placement in translated_placed.items():
            placed_triangles[f_id] = {
                v_id: R @ pos for v_id, pos in placement.items()
            }

        # Check secondary generator 'b' to ensure Im > 0 (upper half-plane)
        if g_b_vec is not None:
            g_b_rot = R @ g_b_vec
            if g_b_rot[1] < 0:
                # Reflection across x-axis if in lower half-plane
                for f_id, placement in placed_triangles.items():
                    for v_id, pos in placement.items():
                        placed_triangles[f_id][v_id] = np.array([pos[0], -pos[1]])

    metadata = {
        "dual_graph": dual_graph,
        "spanning_tree": spanning_tree,
        "holonomy_classes": holonomy_classes,
        "holonomy_transforms": holonomy_transforms,
        "generators": generators,
        "is_hyperbolic": is_hyperbolic,
    }

    return placed_triangles, metadata
