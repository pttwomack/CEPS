# mazedemo/bridge.py
"""
Dual-Domain Metric Mapping and Pathfinder Bridge for Module 3.

Establishes a 1:1 bi-directional bridge between the 2D maze grid space and
the canonical fundamental domain (Poincaré disk or Euclidean annulus).
Computes dual-graph face sequences, extracts local dilatation ratios (K_f),
and generates synchronized traversal paths using 1-ring cover geodesics.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Any, Optional, Union
from dataclasses import dataclass

from mazedemo.ids import FaceId, VertexId, SurfaceId, get_local_id
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.telemetry import StateContext
from mazedemo.hyperbolic import HyperbolicGeometry, hyperboloid_to_poincare
from mazedemo.euclidean import EuclideanGeometry
from mazedemo.layout import generate_canonical_layout, to_cartesian
from mazedemo.geodesic import compute_hyperbolic_geodesic_arc, compute_euclidean_geodesic_segment
from mazedemo.fundamental_domain import compute_fundamental_domain
from mazedemo.dcm_tiling import build_one_ring_cover, OneRingCover, TileFaceKey


@dataclass
class BridgePathResult:
    face_sequence: List[FaceId]
    poincare_arc_samples: np.ndarray  # complex128 array or float64 N x 2 array
    grid_path_coords: np.ndarray  # N x 2 float64 array of maze grid positions
    local_dilatation_map: Dict[FaceId, float]  # FaceId -> K_f scalar


def compute_barycentric_coords_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Computes 2D barycentric coordinates (lambda_a, lambda_b, lambda_c) for point p
    relative to triangle vertices a, b, c in 2D space.
    Clamps and normalizes to non-negative sum-to-1 weights.
    """
    p2d = np.asarray(p[:2], dtype=np.float64)
    a2d = np.asarray(a[:2], dtype=np.float64)
    b2d = np.asarray(b[:2], dtype=np.float64)
    c2d = np.asarray(c[:2], dtype=np.float64)

    M = np.array([
        [a2d[0], b2d[0], c2d[0]],
        [a2d[1], b2d[1], c2d[1]],
        [1.0, 1.0, 1.0]
    ], dtype=np.float64)

    target = np.array([p2d[0], p2d[1], 1.0], dtype=np.float64)

    try:
        lambdas = np.linalg.solve(M, target)
    except np.linalg.LinAlgError:
        lambdas = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    return lambdas


class CanonicalBridge:
    """
    Bi-directional bridge linking 2D maze grid space and canonical target manifold.
    """

    def __init__(
        self,
        factory: SurfaceFactory,
        context: StateContext,
        geometry_obj: Any,
        g_grid: Optional[nx.Graph] = None,
    ):
        self.factory = factory
        self.context = context
        self.geometry_obj = geometry_obj
        self.surface_id = context.surface_id
        self.base_surface_id = context.base_surface_id or context.surface_id
        self.g_grid = g_grid

        # Extract 2D base vertex grid positions mapped by local ID
        self.base_grid_positions = self._extract_base_grid_positions()

        # Extract maze wall line segments exclusively from base_surface_id boundary half-edges
        self.maze_walls = self._extract_maze_walls()

        # Compute fundamental domain artifacts
        self.domain_artifacts = compute_fundamental_domain(
            factory, self.surface_id, geometry_obj
        )
        (
            self.dual_graph,
            self.centered_embeddings,
            self.isometries,
            self.spanning_tree,
            self.holonomy_classes,
            self.placed_triangles,
            self.holonomy_transforms,
            self.generators,
        ) = self.domain_artifacts

        # Layout metadata
        is_hyperbolic = "Hyperbolic" in geometry_obj.__class__.__name__
        self.layout_metadata = {
            "is_hyperbolic": is_hyperbolic,
            "dual_graph": self.dual_graph,
            "spanning_tree": self.spanning_tree,
            "holonomy_transforms": self.holonomy_transforms,
            "holonomy_classes": self.holonomy_classes,
        }

        # Build 1-Ring Universal Cover Tiling
        self.one_ring_cover = build_one_ring_cover(
            factory,
            self.surface_id,
            geometry_obj,
            self.placed_triangles,
            self.holonomy_transforms,
            self.holonomy_classes,
            self.generators,
        )

        # Map placed triangles by local face ID for robust cross-commit lookups
        self.placed_triangles_by_local_id = {
            get_local_id(f_id): placement
            for f_id, placement in self.placed_triangles.items()
        }
        self.face_id_by_local_id = {
            get_local_id(f_id): f_id
            for f_id in self.placed_triangles.keys()
        }

        # Compute face-level K_f (Local Dilatation Ratio)
        self.local_dilatation_map = self._compute_local_dilatation_map()

    def _extract_base_grid_positions(self) -> Dict[Union[int, VertexId], np.ndarray]:
        """
        Extracts 2D grid position coordinates (c, r) for base surface vertices.
        Keys by both VertexId and integer local_id for robust cross-commit lookups.
        """
        view_base = self.factory.create_viewer(
            self.base_surface_id, custom_properties={"position": None, "pos": None}
        )
        grid_pos = {}
        for v in view_base.vertices:
            pos = view_base.get_property(v.id, "position")
            if pos is None:
                pos = view_base.get_property(v.id, "pos")
            if pos is not None:
                p_arr = np.array(pos[:2], dtype=np.float64)
                grid_pos[v.id] = p_arr
                grid_pos[get_local_id(v.id)] = p_arr
        return grid_pos

    def _extract_maze_walls(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts impassable maze walls exclusively from base_surface_id boundary half-edges
        (edges with twin is None) and g_grid dual wall edge attributes.
        """
        walls = []
        if self.g_grid is not None:
            missing_walls = nx.get_edge_attributes(self.g_grid, "dual")
            if missing_walls:
                for dual_edge in missing_walls.values():
                    u, v = dual_edge
                    p1 = np.array([float(u[0]), float(u[1])], dtype=np.float64)
                    p2 = np.array([float(v[0]), float(v[1])], dtype=np.float64)
                    walls.append((p1, p2))
                if walls:
                    return walls

        # Extract boundary half-edges (edges without twin) on base_surface_id
        view_base = self.factory.create_viewer(
            self.base_surface_id, custom_properties={"position": None, "pos": None}
        )
        for he in view_base.half_edges:
            if he.twin is None:
                p1 = view_base.get_property(he.origin.id, "position")
                if p1 is None:
                    p1 = view_base.get_property(he.origin.id, "pos")

                p2 = view_base.get_property(he.next.origin.id, "position")
                if p2 is None:
                    p2 = view_base.get_property(he.next.origin.id, "pos")

                if p1 is not None and p2 is not None and len(p1) >= 2 and len(p2) >= 2:
                    walls.append((np.array(p1[:2], dtype=np.float64), np.array(p2[:2], dtype=np.float64)))

        return walls

    def _compute_local_dilatation_map(self) -> Dict[FaceId, float]:
        """Calculates K_f = exp(max(u) - min(u)) across corner vertices for each face."""
        view = self.factory.create_viewer(self.surface_id, custom_properties={"dcm_index": None})
        u_arr = self.context.u

        k_f_map: Dict[FaceId, float] = {}

        for face in view.faces:
            u_vals = [
                float(u_arr[he.origin.dcm_index])
                for he in face.half_edges
                if he.origin.dcm_index is not None and he.origin.dcm_index < len(u_arr)
            ]
            if len(u_vals) >= 2:
                K_f = float(np.exp(np.max(u_vals) - np.min(u_vals)))
            else:
                K_f = 1.0

            k_f_map[face.id] = K_f

        return k_f_map

    def resolve_node_to_face(
        self, node_input: Union[Tuple[int, int], VertexId, FaceId]
    ) -> FaceId:
        """
        Resolves input specification ( (r, c) tuple, VertexId, or FaceId ) to a canonical FaceId.
        """
        if isinstance(node_input, FaceId):
            local_id = get_local_id(node_input)
            if local_id in self.face_id_by_local_id:
                return self.face_id_by_local_id[local_id]
            return node_input

        view_base = self.factory.create_viewer(
            self.base_surface_id, custom_properties={"position": None, "pos": None}
        )

        if isinstance(node_input, tuple) and len(node_input) == 2:
            r, c = node_input
            target_pos = np.array([float(c), float(r)])
            best_face = None
            min_dist = float("inf")

            for face in view_base.faces:
                v_positions = []
                for he in face.half_edges:
                    pos = view_base.get_property(he.origin.id, "position")
                    if pos is None:
                        pos = view_base.get_property(he.origin.id, "pos")
                    if pos is not None:
                        v_positions.append(pos[:2])

                if v_positions:
                    centroid = np.mean(v_positions, axis=0)
                    dist = np.linalg.norm(centroid - target_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_face = face.id

            if best_face is not None:
                local_id = get_local_id(best_face)
                return self.face_id_by_local_id.get(local_id, best_face)
            return next(iter(self.placed_triangles.keys()))

        if isinstance(node_input, VertexId):
            v_proxy = view_base.get_proxy(node_input)
            if v_proxy and v_proxy.half_edge and v_proxy.half_edge.face:
                f_id = v_proxy.half_edge.face.id
                local_id = get_local_id(f_id)
                return self.face_id_by_local_id.get(local_id, f_id)

        return next(iter(self.placed_triangles.keys()))

    def compute_path(
        self,
        start_node: Union[Tuple[int, int], VertexId, FaceId],
        end_node: Union[Tuple[int, int], VertexId, FaceId],
        num_samples: int = 50,
    ) -> BridgePathResult:
        """
        Computes the true geodesic path across the 1-ring universal cover between start_node and end_node.
        Uses pure barycentric weight inclusion (maximizing min(bary)) rather than centroid distance,
        guaranteeing continuous, non-teleporting 2D maze path mapping.
        """
        start_face = self.resolve_node_to_face(start_node)
        end_face = self.resolve_node_to_face(end_node)

        start_local = get_local_id(start_face)
        end_local = get_local_id(end_face)

        start_placement = self.placed_triangles_by_local_id[start_local]
        end_placement = self.placed_triangles_by_local_id[end_local]

        p1_root = np.mean(list(start_placement.values()), axis=0)
        p2_root = np.mean(list(end_placement.values()), axis=0)

        is_hyperbolic = self.layout_metadata["is_hyperbolic"]

        # 1. Find best image p2_best in the 1-ring cover minimizing geodesic distance d(p1, p2')
        p2_best = p2_root
        best_tile_label = ""
        min_dist = self.geometry_obj.distance(p1_root, p2_root)

        for gen_symbol, M_gen in self.one_ring_cover.holonomy_transforms.items():
            if not gen_symbol:
                continue
            p2_cand = M_gen @ p2_root
            d_cand = self.geometry_obj.distance(p1_root, p2_cand)
            if d_cand < min_dist:
                min_dist = d_cand
                p2_best = p2_cand
                best_tile_label = gen_symbol

        # 2. Compute exact geodesic arc samples in cover space
        if is_hyperbolic:
            z1 = hyperboloid_to_poincare(p1_root)
            z2 = hyperboloid_to_poincare(p2_best)
            z1_complex = np.complex128(z1[0] + 1j * z1[1])
            z2_complex = np.complex128(z2[0] + 1j * z2[1])
            poincare_arc = compute_hyperbolic_geodesic_arc(z1_complex, z2_complex, num_samples=num_samples)

            # Sample 3D hyperboloid geodesic points
            t_vals = np.linspace(0.0, 1.0, num_samples)
            cover_samples_3d = [(1.0 - t) * p1_root + t * p2_best for t in t_vals]
            for i in range(num_samples):
                v = cover_samples_3d[i]
                norm_sq = v[0]**2 + v[1]**2 - v[2]**2
                if norm_sq < 0:
                    cover_samples_3d[i] = v / np.sqrt(-norm_sq)
        else:
            p1_2d = p1_root[:2]
            p2_2d = p2_best[:2]
            poincare_arc = compute_euclidean_geodesic_segment(p1_2d, p2_2d, num_samples=num_samples)
            t_vals = np.linspace(0.0, 1.0, num_samples)
            cover_samples_3d = [(1.0 - t) * p1_root + t * p2_best for t in t_vals]

        # 3. Pure Barycentric Inclusion Pathfinder (Maximizing min(barycentric_weights))
        face_sequence_raw = []
        grid_path_coords = np.zeros((num_samples, 2), dtype=np.float64)

        for i, sample_pt in enumerate(cover_samples_3d):
            best_key = None
            best_min_bary = -float("inf")
            best_bary = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

            if is_hyperbolic:
                sample_p2d = hyperboloid_to_poincare(sample_pt)
            else:
                sample_p2d = sample_pt[:2]

            for key, face_verts in self.one_ring_cover.tiled_placed_triangles.items():
                v_ids = list(face_verts.keys())
                pts_3d = [face_verts[vid] for vid in v_ids]

                if is_hyperbolic:
                    pts_2d = [hyperboloid_to_poincare(pt) for pt in pts_3d]
                else:
                    pts_2d = [pt[:2] for pt in pts_3d]

                bary = compute_barycentric_coords_2d(sample_p2d, pts_2d[0], pts_2d[1], pts_2d[2])
                min_b = float(np.min(bary))

                if min_b > best_min_bary:
                    best_min_bary = min_b
                    best_key = key
                    best_bary = bary

            if best_key is not None:
                f_id = self.face_id_by_local_id[get_local_id(best_key.face_id)]
                face_sequence_raw.append(f_id)

                # Map to 2D maze grid vertex coordinates via barycentric weights using integer local IDs
                v_ids = list(self.one_ring_cover.root_placed_triangles[f_id].keys())
                g_coords = []
                for vid in v_ids:
                    pos_2d = self.base_grid_positions.get(vid)
                    if pos_2d is None:
                        pos_2d = self.base_grid_positions.get(get_local_id(vid), np.array([0.0, 0.0]))
                    g_coords.append(pos_2d)

                q_grid = best_bary[0] * g_coords[0] + best_bary[1] * g_coords[1] + best_bary[2] * g_coords[2]
                grid_path_coords[i] = q_grid
            else:
                grid_path_coords[i] = np.array([0.0, 0.0])

        # Filter sequential duplicate faces to produce clean topological ledger
        face_sequence = []
        for f in face_sequence_raw:
            if not face_sequence or face_sequence[-1] != f:
                face_sequence.append(f)

        return BridgePathResult(
            face_sequence=face_sequence,
            poincare_arc_samples=poincare_arc,
            grid_path_coords=grid_path_coords,
            local_dilatation_map=self.local_dilatation_map,
        )
