import numpy as np
from typing import Optional
from mazedemo.telemetry import StateContext


class SingularityAbortError(Exception):
    """Exception raised when a geometric or topological singularity is hit, necessitating an emergency abort."""

    def __init__(self, message, anomaly_info=None):
        super().__init__(message)
        self.anomaly_info = anomaly_info or {}


class UnclassifiedInvariantViolation(SingularityAbortError):
    """Exception raised when a fundamental mathematical invariant is violated."""

    pass


class InvariantAuditor:
    """Decoupled passive auditor enforcing fundamental mathematical invariants of discrete differential geometry."""

    @staticmethod
    def audit_trial_invariants(lengths_cache: np.ndarray) -> bool:
        """Tier 1 micro-check: Microsecond vectorized check for metric positivity (l > 0)."""
        if lengths_cache is None or len(lengths_cache) == 0:
            return True
        return bool(np.all(np.isfinite(lengths_cache)) and np.all(lengths_cache > 0))

    @staticmethod
    def audit_accepted_invariants(
        context: StateContext,
        boundary_indices: np.ndarray,
        face_corner_vertex_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Tier 2 macro-check: Topological, scale factor, triangle inequality, and Gauss-Bonnet conservation checks on accepted steps."""
        if context is None:
            return

        # 1. Scale Factor Bounds (|u| <= 30)
        u_abs = np.abs(context.u)
        if np.any(u_abs > 30.0):
            violating_indices = np.where(u_abs > 30.0)[0]
            violating_vals = context.u[violating_indices]
            anomaly_info = {
                "type": "scale_factor_limit_exceeded",
                "limit": 30.0,
                "violating_vertices_count": int(len(violating_indices)),
                "sample_violating_indices": violating_indices[:10].tolist(),
                "sample_violating_values": violating_vals[:10].tolist(),
                "geometry": context.geometry,
                "global_distortion_variance": float(np.var(context.u)),
            }
            raise UnclassifiedInvariantViolation(
                "Scale factor exceeded maximum limit of 30.0", anomaly_info
            )

        # 2. Triangle Inequality Violation
        if len(context.half_edge_lengths) > 0:
            lengths = context.half_edge_lengths.reshape(-1, 3)
            l0, l1, l2 = lengths[:, 0], lengths[:, 1], lengths[:, 2]
            valid = (l0 + l1 > l2) & (l1 + l2 > l0) & (l2 + l0 > l1)
            if not np.all(valid):
                invalid_faces = np.where(~valid)[0]
                sample_idx = int(invalid_faces[0])
                anomaly_info = {
                    "type": "triangle_inequality_violation",
                    "violating_faces_count": int(len(invalid_faces)),
                    "sample_violating_face_index": sample_idx,
                    "sample_violating_lengths": lengths[sample_idx].tolist(),
                    "geometry": context.geometry,
                    "global_distortion_variance": float(np.var(context.u)),
                }
                raise UnclassifiedInvariantViolation(
                    "Triangle inequality violated on accepted step", anomaly_info
                )

        # 3. Gauss-Bonnet Conservation (Euclidean total curvature conservation)
        if (
            context.geometry == "euclidean"
            and face_corner_vertex_indices is not None
            and len(face_corner_vertex_indices) > 0
        ):
            n_verts = len(context.u)
            angle_sums = np.bincount(
                face_corner_vertex_indices,
                weights=context.face_angles,
                minlength=n_verts,
            )
            C = np.full(n_verts, 2.0 * np.pi)
            if boundary_indices is not None and boundary_indices.size > 0:
                C[boundary_indices] = np.pi

            K_i = C - angle_sums
            sum_K = float(np.sum(K_i))
            sum_target_K = float(np.sum(context.target_curvature))

            if not np.isclose(sum_K, sum_target_K, atol=1e-4):
                defect_diff = float(np.abs(sum_K - sum_target_K))
                anomaly_info = {
                    "type": "gauss_bonnet_conservation_violation",
                    "sum_K": sum_K,
                    "sum_target_K": sum_target_K,
                    "defect_difference": defect_diff,
                    "geometry": context.geometry,
                    "global_distortion_variance": float(np.var(context.u)),
                }
                raise UnclassifiedInvariantViolation(
                    f"Gauss-Bonnet conservation violated: sum(K)={sum_K:.6f} vs sum(K*)={sum_target_K:.6f}",
                    anomaly_info,
                )
