import numpy as np
import collections
from typing import Dict, Tuple
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from absl import logging
import os
import json
import time

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.refine import (
    refine_euclidean_delaunay,
    refine_ceps_ptolemy,
    refine_hyperbolic_delaunay,
)
from mazedemo.viewer import Viewer, Face, Vertex, HalfEdge
from mazedemo.euclidean import compute_euclidean_angles_from_opposite_lengths
from mazedemo.hyperbolic import (
    compute_hyperbolic_angles_from_opposite_lengths,
    volume_of_ideal_tetrahedron,
    volume_of_hyperideal_tetrahedron,
    lobachevsky,
)
from mazedemo.target_curvature import compute_target_curvatures
from mazedemo.ids import SurfaceId, VertexId
from mazedemo.telemetry import StateContext
from mazedemo.invariant_auditor import (
    InvariantAuditor,
    SingularityAbortError,
    UnclassifiedInvariantViolation,
)
from mazedemo.telemetry_ledger import TelemetryLedger


class DiscreteConformalMapOptimizer:
    """Manages state for the discrete conformal map optimizer."""

    def __init__(
        self,
        factory: SurfaceFactory,
        base_delaunay_id: SurfaceId,
        canonical_hes: list,
        initial_lengths: np.ndarray,
        target_curvature: np.ndarray,
        geometry: str,
    ):
        self.factory = factory
        self.target_curvature = target_curvature
        self.canonical_hes = canonical_hes
        self.geometry = geometry

        # Annotate the surface with a stable 0..N-1 index for NumPy arrays.
        with factory.update_properties(base_delaunay_id) as annotation:
            with annotation.writer() as view:
                for i, v in enumerate(view.vertices):
                    v.dcm_index = i
        self.base_surface_id = annotation.commit("Add dcm_index for optimizer")

        # Trace the canonical half-edges to the new surface with the dcm_index property
        trace_results = factory.trace_elements(
            [self.base_surface_id], set(self.canonical_hes)
        )
        self.canonical_hes = [
            next(iter(trace_results[he_id])) for he_id in self.canonical_hes
        ]

        viewer = factory.create_viewer(
            self.base_surface_id, custom_properties={"dcm_index": None}
        )

        # Validate initial_lengths before taking the log
        if not np.all(np.isfinite(initial_lengths)) or not np.all(initial_lengths > 0):
            raise ValueError(
                "Invalid initial length found. Lengths must be positive and finite."
            )

        self.initial_lengths = initial_lengths.copy()
        if self.geometry == "hyperbolic":
            self.initial_lambda = 2.0 * np.sinh(initial_lengths / 2.0)
        else:
            self.initial_lambda = initial_lengths.copy()

        self.initial_log_lengths = 2.0 * np.log(self.initial_lambda)

        # Update initial surface half-edges to store base Penner coordinates (\lambda)
        with factory.update_properties(self.base_surface_id) as annotation:
            with annotation.writer() as view:
                he_proxy_map = {he.id: he for he in view.half_edges}
                for i, he_id in enumerate(self.canonical_hes):
                    lam = float(self.initial_lambda[i])
                    he = he_proxy_map[he_id]
                    he.length = lam
                    if he.twin:
                        he.twin.length = lam
        self.base_surface_id = annotation.commit("Store initial Penner lengths")

        # Build edge_indices using the stable dcm_index
        edge_indices_list = []
        for he_id in self.canonical_hes:
            he = viewer.get_proxy(he_id)
            v1_idx = he.origin.dcm_index
            v2_idx = he.next.origin.dcm_index
            edge_indices_list.append((v1_idx, v2_idx))
        self.edge_indices = np.array(edge_indices_list)

        self.boundary_indices = np.array(
            [
                v.dcm_index
                for v in viewer.vertices
                if any(he.twin is None for he in v.half_edges)
            ]
        )

        face_corner_list = []
        for face in viewer.faces:
            for he in face.half_edges:
                face_corner_list.append(he.origin.dcm_index)
        self.face_corner_vertex_indices = np.array(face_corner_list, dtype=np.int64)

        self.auditor = InvariantAuditor()

        self.last_u = None
        self.accepted_u = np.zeros(len(viewer.vertices))
        self.accepted_surface_id = self.base_surface_id
        self.current_surface_id = self.base_surface_id
        self.transient_surface_ids = set()
        self.iter_count = 0

        # Safe fallback checkpoints for emergency aborts
        self.last_successful_u = None
        self.last_successful_grad = None
        self.last_successful_surface_id = base_delaunay_id
        self.latest_anomaly = None
        self.anomaly_log = []

        # Telemetry state caches & Phase 3 baseline geometry
        self._current_lengths_cache = np.array([], dtype=np.float64)
        self._current_angles_cache = np.array([], dtype=np.float64)
        self.latest_context = None

        # Build initial_half_edge_lengths per face matching _current_lengths_cache order
        init_lengths_list = []
        for face in viewer.faces:
            hes = face.half_edges
            init_lengths_list.extend([hes[1].length, hes[2].length, hes[0].length])
        self.initial_half_edge_lengths = np.array(init_lengths_list, dtype=np.float64)

        # Build initial graph CSR matrix for FPS landmark selection and distance matrix
        num_verts = len(viewer.vertices)
        graph_rows, graph_cols, graph_data = [], [], []
        for he in viewer.half_edges:
            i = he.origin.dcm_index
            j = he.next.origin.dcm_index
            w = float(he.length)
            graph_rows.extend([i, j])
            graph_cols.extend([j, i])
            graph_data.extend([w, w])

        csr_graph = sp.csc_matrix(
            (graph_data, (graph_rows, graph_cols)), shape=(num_verts, num_verts)
        ).tocsr()

        # Farthest Point Sampling (FPS) for landmark selection
        k_landmarks = min(150, num_verts)
        landmark_list = [0]
        if k_landmarks > 1:
            dist_to_landmarks = dijkstra(csr_graph, directed=False, indices=0)
            while len(landmark_list) < k_landmarks:
                next_landmark = int(np.argmax(dist_to_landmarks))
                landmark_list.append(next_landmark)
                d_next = dijkstra(csr_graph, directed=False, indices=next_landmark)
                dist_to_landmarks = np.minimum(dist_to_landmarks, d_next)

        self.landmark_indices = np.array(landmark_list, dtype=int)
        d_all_landmarks = dijkstra(
            csr_graph, directed=False, indices=self.landmark_indices
        )
        self.initial_landmark_distances = d_all_landmarks[:, self.landmark_indices]

    def compute_vertex_area_weights(
        self, surface_id: SurfaceId = None
    ) -> np.ndarray:
        """Computes normalized dual Voronoi areas w_i for vertices on the given surface."""
        sid = surface_id if surface_id is not None else self.current_surface_id
        if sid is None:
            sid = self.base_surface_id
        view = self.factory.create_viewer(
            sid, custom_properties={"length": None, "dcm_index": None}
        )
        n_verts = len(view.vertices)
        vertex_areas = np.zeros(n_verts, dtype=np.float64)

        for face in view.faces:
            hes = face.half_edges
            lam_lengths = (hes[1].length, hes[2].length, hes[0].length)
            l1, l2, l0 = lam_lengths
            if l1 is None or l2 is None or l0 is None or l1 <= 0 or l2 <= 0 or l0 <= 0:
                continue
            s = 0.5 * (l1 + l2 + l0)
            area_sq = s * (s - l1) * (s - l2) * (s - l0)
            f_area = np.sqrt(max(0.0, float(area_sq)))
            third_area = f_area / 3.0
            for he in hes:
                vertex_areas[he.origin.dcm_index] += third_area

        total_area = np.sum(vertex_areas)
        if total_area > 1e-12:
            return vertex_areas / total_area
        else:
            return np.ones(n_verts, dtype=np.float64) / float(n_verts)


    def update_state_context(
        self, iteration: int, u: np.ndarray, grad: np.ndarray, energy: float
    ) -> StateContext:
        """Instantiates and updates self.latest_context with current cached state, running Tier 2 invariant checks."""
        self.latest_context = StateContext(
            iteration=iteration,
            surface_id=self.current_surface_id,
            geometry=self.geometry,
            u=u.copy(),
            gradient=grad.copy(),
            energy=energy,
            half_edge_lengths=self._current_lengths_cache.copy(),
            face_angles=self._current_angles_cache.copy(),
            target_curvature=self.target_curvature.copy(),
            initial_half_edge_lengths=self.initial_half_edge_lengths.copy(),
            landmark_indices=self.landmark_indices.copy(),
            initial_landmark_distances=self.initial_landmark_distances.copy(),
            base_surface_id=self.base_surface_id,
        )
        
        InvariantAuditor.audit_accepted_invariants(
            self.latest_context,
            self.boundary_indices,
            self.face_corner_vertex_indices,
        )
        return self.latest_context

    def commit_accepted_state(self, u: np.ndarray, surface_id: SurfaceId):
        """Commits an accepted state so future trial steps scale from this state."""
        self.accepted_u = u.copy()
        self.accepted_surface_id = surface_id
        self.anomaly_log.clear()
        self.latest_anomaly = None

    def _update_and_refine_surface(self, u: np.ndarray):
        """Updates and refines the surface for a given u relative to the last accepted state."""
        if self.last_u is not None and np.allclose(u, self.last_u, rtol=0, atol=1e-14):
            return

        refined_id = None
        known_commits = set(self.factory._commits.keys())

        # Scale edge lengths from the accepted Delaunay surface using delta_u = u - accepted_u
        if self.accepted_surface_id is not None:
            try:
                delta_u = u - self.accepted_u
                has_none = False
                with self.factory.update_properties(self.accepted_surface_id) as annotation:
                    with annotation.writer() as view:
                        processed_edges = set()
                        for he in view.half_edges:
                            if he.twin:
                                edge_key = frozenset({he.id, he.twin.id})
                                if edge_key in processed_edges:
                                    continue
                                processed_edges.add(edge_key)

                            if he.length is None or he.length <= 0:
                                has_none = True
                                break

                            v1_idx = int(he.origin.dcm_index)
                            v2_idx = int(he.next.origin.dcm_index)
                            scale = float(np.exp((delta_u[v1_idx] + delta_u[v2_idx]) / 2.0))
                            new_len = float(he.length * scale)

                            he.length = new_len
                            if he.twin:
                                he.twin.length = new_len

                if not has_none:
                    temp_id = annotation.commit("Incremental metric update")
                    refined_id = refine_ceps_ptolemy(self.factory, temp_id)
            except Exception:
                refined_id = None

        # Fallback to base_surface_id if incremental update fails
        if refined_id is None:
            with self.factory.update_properties(self.base_surface_id) as annotation:
                with annotation.writer() as new_view:
                    he_proxy_map = {he.id: he for he in new_view.half_edges}
                    for i, he_id in enumerate(self.canonical_hes):
                        idx1 = int(self.edge_indices[i, 0])
                        idx2 = int(self.edge_indices[i, 1])
                        new_log_lengths_i = float(
                            self.initial_log_lengths[i]
                            + u[idx1]
                            + u[idx2]
                        )
                        new_len = float(np.exp(new_log_lengths_i / 2.0))
                        he = he_proxy_map[he_id]
                        he.length = new_len
                        if he.twin:
                            he.twin.length = new_len
                temp_id = annotation.commit("Full metric update for refinement")

            refined_id = refine_ceps_ptolemy(self.factory, temp_id)

        self.current_surface_id = refined_id
        self.last_u = u.copy() if refined_id is not None else None

        # Update face_corner_vertex_indices for the refined surface
        if self.current_surface_id is not None:
            view = self.factory.create_viewer(
                self.current_surface_id, custom_properties={"dcm_index": None}
            )
            face_corner_list = []
            for face in view.faces:
                for he in face.half_edges:
                    face_corner_list.append(he.origin.dcm_index)
            self.face_corner_vertex_indices = np.array(face_corner_list, dtype=np.int64)

        # Record transient commits created during refinement
        new_commits = set(self.factory._commits.keys()) - known_commits
        self.transient_surface_ids.update(new_commits)
        if self.current_surface_id:
            self.transient_surface_ids.add(self.current_surface_id)

    def _compute_energy_at_state(
        self, surface_id: SurfaceId, u: np.ndarray
    ) -> Dict[str, float] | float:
        """A 'pure' energy calculation that does not refine the mesh."""
        if surface_id is None:
            return np.inf

        # Emergency Abort Check: extreme scale factors trigger SingularityAbortError
        if np.any(np.abs(u) > 30.0):
            violating_indices = np.where(np.abs(u) > 30.0)[0]
            violating_vals = u[violating_indices]
            anomaly_info = {
                "type": "scale_factor_limit_exceeded",
                "limit": 30.0,
                "violating_vertices_count": len(violating_indices),
                "sample_violating_indices": violating_indices[:10].tolist(),
                "sample_violating_values": violating_vals[:10].tolist(),
                "geometry": self.geometry,
                "global_distortion_variance": float(np.var(u)),
            }
            raise SingularityAbortError(
                "Scale factor exceeded maximum limit of 30.0", anomaly_info
            )

        try:
            view = self.factory.create_viewer(
                surface_id, custom_properties={"length": None, "dcm_index": None}
            )

            linear_term = 0.0
            interior_indices = np.setdiff1d(np.arange(len(u)), self.boundary_indices)
            linear_term += np.sum(u[interior_indices]) * 2 * np.pi
            if self.boundary_indices.size > 0:
                linear_term += np.sum(u[self.boundary_indices]) * np.pi
            linear_term -= np.dot(u, self.target_curvature)

            pi_term = 0.0
            angle_lambda_term = 0.0
            volume_term = 0.0
            lengths_list = []
            angles_list = []

            processed_edges = set()
            for he in view.half_edges:
                if he.length is None or he.length <= 0:
                    return np.inf

                v1 = int(he.origin.dcm_index)
                v2 = int(he.next.origin.dcm_index)

                if self.geometry == "hyperbolic":
                    lambda_tilde = 2 * np.log(he.length / 2.0)
                    lambda_0 = lambda_tilde - u[v1] - u[v2]
                else:
                    lambda_tilde = 2 * np.log(he.length)
                    lambda_0 = lambda_tilde

                if he.twin:
                    edge_key = frozenset({he.id, he.twin.id})
                    if edge_key not in processed_edges:
                        processed_edges.add(edge_key)
                        pi_term -= np.pi * lambda_0
                else:
                    pi_term -= (np.pi / 2.0) * lambda_0

            for face in view.faces:
                hes = face.half_edges
                if any(he.length is None for he in hes):
                    return np.inf

                lam_lengths = (hes[1].length, hes[2].length, hes[0].length)

                try:
                    if self.geometry == "hyperbolic":
                        finite_lengths = tuple(
                            2.0 * np.arcsinh(l / 2.0) for l in lam_lengths
                        )
                        angles = compute_hyperbolic_angles_from_opposite_lengths(
                            finite_lengths
                        )
                        a, b, c = angles
                        alpha_23 = 0.5 * (np.pi + a - b - c)
                        alpha_31 = 0.5 * (np.pi - a + b - c)
                        alpha_12 = 0.5 * (np.pi - a - b + c)

                        vol = 2 * 0.5 * (
                            lobachevsky(a)
                            + lobachevsky(b)
                            + lobachevsky(c)
                            + lobachevsky(alpha_12)
                            + lobachevsky(alpha_23)
                            + lobachevsky(alpha_31)
                            + lobachevsky(0.5 * (np.pi - a - b - c))
                        )
                        lambda_tildes = [
                            2 * np.log(he.length / 2.0)
                            for he in (hes[1], hes[2], hes[0])
                        ]
                        u_face_v1 = [
                            he.origin.dcm_index for he in (hes[1], hes[2], hes[0])
                        ]
                        u_face_v2 = [
                            he.next.origin.dcm_index for he in (hes[1], hes[2], hes[0])
                        ]
                        lambda_0s = [
                            lt - u[v1] - u[v2]
                            for lt, v1, v2 in zip(lambda_tildes, u_face_v1, u_face_v2)
                        ]

                        # alpha_ij * lambda_0_ij
                        angle_lambda_term += (
                            alpha_23 * lambda_0s[0]
                            + alpha_31 * lambda_0s[1]
                            + alpha_12 * lambda_0s[2]
                        )
                        # alpha_i * lambda_i where lambda_i = -u_i
                        u_face = [
                            u[he.origin.dcm_index] for he in (hes[0], hes[1], hes[2])
                        ]
                        angle_lambda_term += sum(
                            ang * (-u_v) for ang, u_v in zip(angles, u_face)
                        )
                    else:
                        angles = compute_euclidean_angles_from_opposite_lengths(
                            lam_lengths
                        )
                        vol = 2 * volume_of_ideal_tetrahedron(angles)
                        lambda_tildes = [
                            2 * np.log(he.length) for he in (hes[1], hes[2], hes[0])
                        ]
                        angle_lambda_term += sum(
                            a * l for a, l in zip(angles, lambda_tildes)
                        )

                    volume_term += vol
                    if self.geometry == "hyperbolic":
                        lengths_list.extend(finite_lengths)
                    else:
                        lengths_list.extend(lam_lengths)
                    angles_list.extend(angles)
                except (ValueError, ZeroDivisionError) as e:
                    v_ids = [he.origin.id for he in hes]
                    v_dcm_indices = [he.origin.dcm_index for he in hes]
                    u_vals = [u[idx] for idx in v_dcm_indices]
                    lengths_list_err = [float(l) for l in lam_lengths]
                    anomaly_entry = {
                        "type": "triangle_inequality_violation",
                        "exception": type(e).__name__,
                        "message": str(e),
                        "face_id": str(face.id),
                        "vertex_ids": [str(vid) for vid in v_ids],
                        "vertex_dcm_indices": v_dcm_indices,
                        "vertex_scale_factors": u_vals,
                        "half_edge_lengths": lengths_list_err,
                        "geometry": self.geometry,
                        "global_distortion_variance": float(np.var(u)),
                    }
                    self.latest_anomaly = anomaly_entry
                    self.anomaly_log.append(anomaly_entry)
                    return np.inf

            if not np.isfinite(volume_term) or not np.isfinite(angle_lambda_term):
                return np.inf

            self._current_lengths_cache = np.array(lengths_list, dtype=np.float64)
            self._current_angles_cache = np.array(angles_list, dtype=np.float64)

            # Tier 1 micro-check: Metric positivity (l > 0)
            if not InvariantAuditor.audit_trial_invariants(self._current_lengths_cache):
                return np.inf

            return {
                "linear": linear_term,
                "pi": pi_term,
                "angle_lambda": angle_lambda_term,
                "volume": volume_term,
            }
        except Exception as e:
            if isinstance(e, SingularityAbortError):
                raise e
            return np.inf

    def energy(self, u: np.ndarray) -> float:
        """The energy functional for the current conformal state."""
        try:
            self._update_and_refine_surface(u)

            if self.current_surface_id is None:
                anomaly_info = {
                    "type": "refinement_failed",
                    "geometry": self.geometry,
                    "global_distortion_variance": float(np.var(u)),
                }
                raise SingularityAbortError(
                    "Mesh refinement failed or stalled topologically during energy evaluation",
                    anomaly_info,
                )

            energy_val = self._compute_energy_at_state(self.current_surface_id, u)

            if not isinstance(energy_val, dict):
                return np.inf

            return (
                energy_val["linear"]
                + energy_val["pi"]
                + energy_val["angle_lambda"]
                + energy_val["volume"]
            )
        except Exception as e:
            if isinstance(e, SingularityAbortError):
                raise e
            return np.inf

    def _compute_gradient_at_state(
        self, surface_id: SurfaceId
    ) -> Dict[str, np.ndarray]:
        """A 'pure' gradient calculation that does not refine the mesh."""
        if surface_id is None:
            return {
                "linear": np.zeros(self.target_curvature.shape),
                "volume": np.full(self.target_curvature.shape, 1e6),
            }
        view = self.factory.create_viewer(
            surface_id, custom_properties={"length": None}
        )
        angle_sums = np.zeros(len(view.vertices))
        for face in view.faces:
            hes = face.half_edges
            if any(he.length is None for he in hes):
                raise ValueError("Missing half-edge length during gradient computation.")

            lam_lengths = (hes[1].length, hes[2].length, hes[0].length)

            if self.geometry == "euclidean":
                angles = compute_euclidean_angles_from_opposite_lengths(lam_lengths)
            elif self.geometry == "hyperbolic":
                finite_lengths = tuple(
                    2.0 * np.arcsinh(l / 2.0) for l in lam_lengths
                )
                angles = compute_hyperbolic_angles_from_opposite_lengths(finite_lengths)
            else:
                raise NotImplementedError(
                    f"Geometry '{self.geometry}' not supported for gradient calculation."
                )

            for i, he in enumerate(hes):
                try:
                    dcm_idx = he.origin.dcm_index
                    angle_sums[dcm_idx] += angles[i]
                except AttributeError:
                    raise ValueError(f"Could not find dcm_index for vertex {he.origin.id}")

        C = np.full_like(angle_sums, 2 * np.pi)
        if self.boundary_indices.size > 0:
            C[self.boundary_indices] = np.pi

        linear_part = C - self.target_curvature
        volume_part = -angle_sums
        return {"linear": linear_part, "volume": volume_part}

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Computes the exact gradient of the discrete conformal energy functional."""
        if np.any(np.abs(u) > 30.0):
            violating_indices = np.where(np.abs(u) > 30.0)[0]
            violating_vals = u[violating_indices]
            anomaly_info = {
                "type": "scale_factor_limit_exceeded",
                "limit": 30.0,
                "violating_vertices_count": len(violating_indices),
                "sample_violating_indices": violating_indices[:10].tolist(),
                "sample_violating_values": violating_vals[:10].tolist(),
                "geometry": self.geometry,
                "global_distortion_variance": float(np.var(u)),
            }
            raise SingularityAbortError(
                "Scale factor exceeded maximum limit of 30.0", anomaly_info
            )

        try:
            self._update_and_refine_surface(u)

            if self.current_surface_id is None:
                anomaly_info = {
                    "type": "refinement_failed",
                    "geometry": self.geometry,
                    "global_distortion_variance": float(np.var(u)),
                }
                raise SingularityAbortError(
                    "Mesh refinement failed or stalled topologically during gradient evaluation",
                    anomaly_info,
                )

            grad_terms = self._compute_gradient_at_state(self.current_surface_id)
            grad = grad_terms["linear"] + grad_terms["volume"]

            # Checkpoint the successful state
            self.last_successful_u = u.copy()
            self.last_successful_grad = grad.copy()
            self.last_successful_surface_id = self.current_surface_id

            return grad
        except Exception as e:
            if isinstance(e, SingularityAbortError):
                raise e
            return np.full(self.target_curvature.shape, 1e6)

    def jacobian(self, u: np.ndarray) -> sp.csc_matrix:
        """Computes the cotan-Laplacian on the current ideal triangulation."""
        if np.any(np.abs(u) > 30.0):
            violating_indices = np.where(np.abs(u) > 30.0)[0]
            violating_vals = u[violating_indices]
            anomaly_info = {
                "type": "scale_factor_limit_exceeded",
                "limit": 30.0,
                "violating_vertices_count": len(violating_indices),
                "sample_violating_indices": violating_indices[:10].tolist(),
                "sample_violating_values": violating_vals[:10].tolist(),
                "geometry": self.geometry,
                "global_distortion_variance": float(np.var(u)),
            }
            raise SingularityAbortError(
                "Scale factor exceeded maximum limit of 30.0", anomaly_info
            )

        if self.current_surface_id is None:
            anomaly_info = {
                "type": "refinement_failed",
                "geometry": self.geometry,
                "global_distortion_variance": float(np.var(u)),
            }
            raise SingularityAbortError(
                "Mesh refinement failed or stalled topologically during Jacobian evaluation",
                anomaly_info,
            )

        try:
            view = self.factory.create_viewer(
                self.current_surface_id,
                custom_properties={"length": None, "dcm_index": None},
            )
            n_verts = len(view.vertices)

            if self.geometry == "hyperbolic":
                # Hyperbolic exact cotan-Laplacian Hessian
                w_per_he = {}
                for face in view.faces:
                    hes = face.half_edges
                    lam_lengths = (hes[1].length, hes[2].length, hes[0].length)
                    finite_lengths = tuple(
                        2.0 * np.arcsinh(l / 2.0) for l in lam_lengths
                    )
                    angles = compute_hyperbolic_angles_from_opposite_lengths(finite_lengths)
                    arg0 = 0.5 * (np.pi - angles[0] - angles[1] + angles[2])
                    arg1 = 0.5 * (np.pi + angles[0] - angles[1] - angles[2])
                    arg2 = 0.5 * (np.pi - angles[0] + angles[1] - angles[2])

                    if np.sin(arg0) > 1e-9:
                        w_per_he[hes[0].id] = 1.0 / np.tan(arg0)
                    if np.sin(arg1) > 1e-9:
                        w_per_he[hes[1].id] = 1.0 / np.tan(arg1)
                    if np.sin(arg2) > 1e-9:
                        w_per_he[hes[2].id] = 1.0 / np.tan(arg2)

                rows, cols, data = [], [], []
                diag_vals = np.zeros(n_verts)
                processed_edges = set()
                for he in view.half_edges:
                    i = he.origin.dcm_index
                    j = he.next.origin.dcm_index
                    edge_tuple = tuple(sorted((i, j)))
                    if edge_tuple in processed_edges:
                        continue
                    processed_edges.add(edge_tuple)

                    w_ij = 0.5 * w_per_he.get(he.id, 0)
                    if he.twin:
                        w_ij += 0.5 * w_per_he.get(he.twin.id, 0)

                    l_finite = float(2.0 * np.arcsinh(he.length / 2.0))
                    T_ij = np.tanh(l_finite / 2.0) ** 2
                    val = -w_ij * (1.0 - T_ij)
                    if np.isfinite(val):
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([val, val])
                        diag_vals[i] += w_ij * (1.0 + T_ij)
                        diag_vals[j] += w_ij * (1.0 + T_ij)

                rows.extend(range(n_verts))
                cols.extend(range(n_verts))
                data.extend(diag_vals)
                return sp.csc_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
            else:
                cotangents_per_he = {}
                for face in view.faces:
                    hes = face.half_edges
                    try:
                        lengths = (hes[1].length, hes[2].length, hes[0].length)
                        angles = compute_euclidean_angles_from_opposite_lengths(
                            lengths
                        )
                        if np.sin(angles[2]) > 1e-9:
                            cotangents_per_he[hes[0].id] = 1.0 / np.tan(angles[2])
                        if np.sin(angles[0]) > 1e-9:
                            cotangents_per_he[hes[1].id] = 1.0 / np.tan(angles[0])
                        if np.sin(angles[1]) > 1e-9:
                            cotangents_per_he[hes[2].id] = 1.0 / np.tan(angles[1])
                    except (ValueError, AttributeError):
                        raise ValueError(
                            "Jacobian calculation failed due to missing edge lengths."
                        )

                rows, cols, data = [], [], []
                diag_vals = np.zeros(n_verts)

                processed_edges = set()
                for he in view.half_edges:
                    try:
                        i = he.origin.dcm_index
                        j = he.next.origin.dcm_index
                    except AttributeError:
                        raise ValueError(
                            f"Could not find dcm_index for vertices of edge associated with {he.id}"
                        )

                    edge_tuple = tuple(sorted((i, j)))
                    if edge_tuple in processed_edges:
                        continue
                    processed_edges.add(edge_tuple)

                    cot_sum = cotangents_per_he.get(he.id, 0)
                    if he.twin:
                        cot_sum += cotangents_per_he.get(he.twin.id, 0)

                    val = -0.5 * cot_sum
                    if np.isfinite(val):
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([val, val])
                        diag_vals[i] -= val
                        diag_vals[j] -= val

                rows.extend(range(n_verts))
                cols.extend(range(n_verts))
                data.extend(diag_vals)

                return sp.csc_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        except Exception as e:
            if isinstance(e, SingularityAbortError):
                raise e
            n_verts = len(self.target_curvature)
            return sp.eye(n_verts, format="csc") * 1e6


def project_zero_mean_area(x: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """Projects vector x into the zero-mean subspace."""
    if w is None:
        return x - np.mean(x)
    return x - np.dot(w, x)


def discrete_conformal_map(
    factory: SurfaceFactory,
    surface_id: SurfaceId,
    target_curvature: np.ndarray = None,
    return_gradient=False,
    geometry: str = "euclidean",
    return_defect=False,
    return_context=False,
):
    """Finds a conformally equivalent metric using custom Newton-Raphson solver with Armijo backtracking line search, vertex pinning (Euclidean), step capping, and gradient descent fallback."""
    if geometry == "hyperbolic":
        delaunay_surface_id = refine_hyperbolic_delaunay(factory, surface_id)
    else:
        delaunay_surface_id = refine_euclidean_delaunay(factory, surface_id)

    viewer = factory.create_viewer(
        delaunay_surface_id, custom_properties={"length": None}
    )

    if target_curvature is None:
        if geometry == "euclidean":
            target_curvature, _ = compute_target_curvatures(factory, delaunay_surface_id)
        else:
            target_curvature = np.zeros(len(viewer.vertices))

    canonical_hes = []
    initial_lengths_list = []
    processed_edges = set()
    for he in viewer.half_edges:
        if he.twin:
            edge_key = frozenset({he.id, he.twin.id})
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)

        try:
            length = he.length
            if length is None:
                raise ValueError(
                    f"Missing initial length for edge associated with {he.id}"
                )
        except AttributeError:
            raise ValueError(f"Missing initial length for edge associated with {he.id}")

        canonical_hes.append(he.id)
        initial_lengths_list.append(he.length)

    initial_lengths_np = np.array(initial_lengths_list)
    optimizer = DiscreteConformalMapOptimizer(
        factory,
        delaunay_surface_id,
        canonical_hes,
        initial_lengths_np,
        target_curvature,
        geometry=geometry,
    )

    start_time = time.time()
    n_verts = len(viewer.vertices)
    u = np.zeros(n_verts)

    if geometry == "euclidean":
        u = project_zero_mean_area(u)

    max_iter = 100
    tol = 1e-5
    converged = False
    message = ""
    anomaly_tracking = None

    ledger = TelemetryLedger(geometry=geometry)

    try:
        # Initialize StateContext at iteration 0 (initial state)
        init_energy = optimizer.energy(u)
        init_grad = optimizer.gradient(u)
        optimizer.update_state_context(
            iteration=0, u=u, grad=init_grad, energy=init_energy
        )
        ledger.stream_iteration(optimizer.latest_context)

        for iteration in range(max_iter):
            if geometry == "euclidean":
                u = project_zero_mean_area(u)

            current_energy = optimizer.energy(u)
            g = optimizer.gradient(u)
            optimizer.update_state_context(
                iteration=iteration, u=u, grad=g, energy=current_energy
            )
            ledger.stream_iteration(optimizer.latest_context)

            g_norm = np.linalg.norm(g)
            max_defect = np.max(np.abs(g))

            if g_norm < tol or max_defect < tol:
                converged = True
                message = f"Converged successfully in {iteration} iterations."
                ledger.stream_iteration(optimizer.latest_context, exit_message=message)
                break

            H = optimizer.jacobian(u)

            p_newton = None
            try:
                if geometry == "euclidean":
                    w = np.ones(n_verts) / float(n_verts)
                    w_col = sp.csc_matrix(w.reshape(-1, 1))
                    w_row = sp.csc_matrix(w.reshape(1, -1))
                    zero_corner = sp.csc_matrix((1, 1))
                    H_bordered = sp.bmat([[H, w_col], [w_row, zero_corner]], format="csc")
                    rhs = np.append(-g, 0.0)
                    try:
                        sol = spsolve(H_bordered, rhs)
                        p_newton = sol[:n_verts]
                    except Exception:
                        reg = sp.eye(n_verts, format="csc") * 1e-11
                        H_bordered_reg = sp.bmat([[H + reg, w_col], [w_row, zero_corner]], format="csc")
                        sol = spsolve(H_bordered_reg, rhs)
                        p_newton = sol[:n_verts]

                    if p_newton is not None and np.all(np.isfinite(p_newton)):
                        p_newton = project_zero_mean_area(p_newton)
                    else:
                        p_newton = None
                else:
                    # Fix 5: Attempt pure spsolve first in hyperbolic space
                    try:
                        p_newton = spsolve(H, -g)
                    except Exception:
                        reg = sp.eye(n_verts, format="csc") * 1e-11
                        H_solvable = H + reg
                        p_newton = spsolve(H_solvable, -g)
            except Exception as e:
                anomaly_entry = {
                    "type": "hessian_factorization_error",
                    "exception": type(e).__name__,
                    "message": str(e),
                    "iteration": iteration,
                    "geometry": geometry,
                    "global_distortion_variance": float(np.var(u)),
                }
                optimizer.latest_anomaly = anomaly_entry
                optimizer.anomaly_log.append(anomaly_entry)
                logging.warning(f"Newton solve failed on iteration {iteration}: {e}")
                p_newton = None

            # Fix 2: Strict Descent Direction Validation (g^T p_newton < 0)
            if p_newton is not None and np.all(np.isfinite(p_newton)):
                g_p = float(np.dot(g, p_newton))
                if g_p >= 0.0:
                    p_newton = None
            else:
                p_newton = None

            # Cap max step in infinity norm to prevent scale factor overflow
            max_u_step = 0.5 if geometry == "hyperbolic" else 1.0
            if p_newton is not None:
                max_step = np.max(np.abs(p_newton))
                if max_step > max_u_step and max_step > 0:
                    p_newton = (max_u_step / max_step) * p_newton

            initial_energy = current_energy
            if not np.isfinite(initial_energy):
                raise SingularityAbortError("Initial energy evaluation is non-finite.")

            step_accepted = False
            best_u = u.copy()
            c1 = 1e-4

            accepted_energy = None

            # 1. Try Newton step with Armijo backtracking (Fix 3 & Fix 4)
            if p_newton is not None:
                g_p = float(np.dot(g, p_newton))
                alpha_ls = 1.0
                for ls_step in range(12):
                    # Pure step parallel to p_newton without modifying direction during cuts
                    u_cand = u + alpha_ls * p_newton

                    try:
                        cand_energy = optimizer.energy(u_cand)
                        # Fix 3: Strict energy reduction check with negative directional derivative
                        if np.isfinite(cand_energy) and cand_energy <= initial_energy + c1 * alpha_ls * g_p:
                            best_u = u_cand
                            accepted_energy = cand_energy
                            step_accepted = True
                            break
                    except (SingularityAbortError, ValueError, ZeroDivisionError):
                        pass

                    alpha_ls *= 0.5

            # 2. Gradient Descent Fallback (-g) if Newton step was not accepted
            if not step_accepted:
                if geometry == "euclidean":
                    p_grad = project_zero_mean_area(-g)
                else:
                    p_grad = -g.copy()

                g_p_grad = float(np.dot(g, p_grad))
                if g_p_grad >= 0.0:
                    p_grad = -g.copy()
                    g_p_grad = float(np.dot(g, p_grad))

                max_step_g = np.max(np.abs(p_grad))
                if max_step_g > max_u_step and max_step_g > 0:
                    p_grad = (max_u_step / max_step_g) * p_grad
                    g_p_grad = float(np.dot(g, p_grad))

                alpha_ls = 1.0
                for ls_step in range(12):
                    u_cand = u + alpha_ls * p_grad

                    try:
                        cand_energy = optimizer.energy(u_cand)
                        if np.isfinite(cand_energy) and cand_energy <= initial_energy + c1 * alpha_ls * g_p_grad:
                            best_u = u_cand
                            accepted_energy = cand_energy
                            step_accepted = True
                            logging.info(
                                f"Iteration {iteration}: Gradient descent fallback accepted step."
                            )
                            break
                    except (SingularityAbortError, ValueError, ZeroDivisionError):
                        pass

                    alpha_ls *= 0.5

            if not step_accepted:
                logging.warning("Line search failed to find acceptable energy reduction step.")
                message = "Line search stalled."
                ledger.stream_iteration(optimizer.latest_context, exit_message=message)
                break

            u = best_u
            if geometry == "euclidean":
                u = project_zero_mean_area(u)

            optimizer.commit_accepted_state(u, optimizer.current_surface_id)
            current_energy = accepted_energy if accepted_energy is not None else optimizer.energy(u)
            current_grad = optimizer.gradient(u)
            optimizer.update_state_context(
                iteration=iteration + 1, u=u, grad=current_grad, energy=current_energy
            )
            ledger.stream_iteration(optimizer.latest_context)

        if not converged and not message:
            message = f"Reached maximum iteration limit ({max_iter})."
            ledger.stream_iteration(optimizer.latest_context, exit_message=message)

        final_id = optimizer.current_surface_id
        if not final_id:
            raise RuntimeError("Optimization failed to produce a valid surface.")

    except SingularityAbortError as sae:
        converged = False
        u = optimizer.last_successful_u if optimizer.last_successful_u is not None else u
        final_grad = (
            optimizer.last_successful_grad
            if optimizer.last_successful_grad is not None
            else np.full(u.shape, 1e6)
        )
        final_id = (
            optimizer.last_successful_surface_id
            if optimizer.last_successful_surface_id is not None
            else delaunay_surface_id
        )
        reason_str = str(sae)
        if isinstance(sae, UnclassifiedInvariantViolation):
            message = f"Emergency Abort [UNCLASSIFIED_INVARIANT_VIOLATION]: {reason_str}"
        else:
            message = f"Emergency Abort: {reason_str}"
        anomaly_tracking = sae.anomaly_info

        ledger.log_crash(
            exception=sae,
            anomaly_info=anomaly_tracking,
            context=optimizer.latest_context,
            fallback_u=u,
            fallback_grad=final_grad,
        )

        print("\n=============================================")
        print(f"!!! EMERGENCY ABORT DETECTED ({geometry.upper()}) !!!")
        print(f"Reason: {message}")
        if anomaly_tracking:
            print("Anomaly Details:")
            for k, v in anomaly_tracking.items():
                print(f"  {k}: {v}")
        print("=============================================\n")

    end_time = time.time()
    total_time = end_time - start_time

    ctx = optimizer.latest_context
    if ctx is not None:
        final_u = ctx.u
        final_grad_flat = ctx.gradient.ravel()
    else:
        final_u = u
        final_grad_flat = np.asarray(final_grad if final_grad is not None else np.zeros(n_verts)).ravel()

    grad_norm = float(np.linalg.norm(final_grad_flat))
    max_residual_defect = float(np.max(np.abs(final_grad_flat)))
    mean_residual_defect = float(np.mean(np.abs(final_grad_flat)))
    global_distortion_variance = float(np.var(final_u))
    max_scale_factor = float(np.max(np.abs(final_u)))
    max_scale_vertex = int(np.argmax(np.abs(final_u)))

    if grad_norm < tol or max_residual_defect < tol:
        converged = True

    print("=============================================")
    print("Discrete Conformal Map Diagnostics:")
    print(f"  Geometry:                   {geometry.upper()}")
    print(f"  Converged:                  {converged}")
    print(f"  Gradient Norm:              {grad_norm:.6e}")
    print(f"  Max Residual Angle Defect:  {max_residual_defect:.6e}")
    print(f"  Mean Residual Angle Defect: {mean_residual_defect:.6e}")
    print(f"  Max Scale Factor (|u|):     {max_scale_factor:.4f} (Vertex {max_scale_vertex})")
    print(f"  Global Distortion Variance: {global_distortion_variance:.6e}")
    print(f"  Total Time:                 {total_time:.2f} seconds")
    print(f"  Message:                    {message}")
    print("=============================================\n")

    defect_data = {
        "max_residual_angle_defect": max_residual_defect,
        "mean_residual_angle_defect": mean_residual_defect,
        "gradient_norm": grad_norm,
        "global_distortion_variance": global_distortion_variance,
        "max_scale_factor": max_scale_factor,
        "max_scale_vertex_index": max_scale_vertex,
        "total_time_seconds": total_time,
        "converged": converged,
        "message": message,
        "timestamp": time.time(),
    }
    if ctx is not None:
        try:
            from mazedemo.distortion_auditor import audit_distortion
            scorecard = audit_distortion(ctx, factory=factory)
            defect_data["delta_M_scorecard"] = scorecard
        except Exception as e:
            logging.warning(f"Could not compute delta_M_scorecard: {e}")

    if anomaly_tracking:
        defect_data["anomaly_tracking"] = anomaly_tracking

    try:
        with open("dcm_diagnostics.json", "w") as f:
            json.dump(defect_data, f, indent=4)
    except Exception as e:
        logging.warning(f"Could not save dcm_diagnostics.json: {e}")

    if os.path.exists("/content/drive/MyDrive/"):
        gdrive_dir = "/content/drive/MyDrive/mazetool_results/"
        try:
            os.makedirs(gdrive_dir, exist_ok=True)
            dest_path = os.path.join(gdrive_dir, "residual_defects.json")
            with open(dest_path, "w") as f:
                json.dump(defect_data, f, indent=4)
            logging.info(f"Saved residual defects to Google Drive: {dest_path}")
        except Exception as e:
            logging.warning(
                f"Could not save to Google Drive due to: {e}. Falling back to local save."
            )
            with open("residual_defects.json", "w") as f:
                json.dump(defect_data, f, indent=4)
    else:
        try:
            with open("residual_defects.json", "w") as f:
                json.dump(defect_data, f, indent=4)
        except Exception as e:
            logging.warning(f"Could not save residual_defects.json locally: {e}")

    with factory.update_properties(final_id) as annotation:
        with annotation.writer() as view:
            for v in view.vertices:
                v.residual_angle_defect = float(np.abs(final_grad_flat[int(v.dcm_index)]))
            if geometry == "hyperbolic":
                for he in view.half_edges:
                    if he.length is not None:
                        he.length = float(2.0 * np.arcsinh(he.length / 2.0))
    final_id = annotation.commit("Store residual angle defects and finite lengths")

    if return_context:
        return final_id, optimizer.latest_context
    if return_gradient:
        return final_id, final_grad_flat
    if return_defect:
        return final_id, max_residual_defect
    return final_id
