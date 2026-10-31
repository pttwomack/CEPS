from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any
from mazedemo.ids import SurfaceId, FaceId


@dataclass(slots=True)
class StateContext:
    """Lightweight in-memory cache populated directly during solver passes.

    Decouples optimization steps from geometric checks and telemetry, caching
    flat NumPy arrays to avoid redundant trigonometric and topological evaluations.
    """

    iteration: int
    surface_id: SurfaceId
    geometry: str
    u: np.ndarray
    gradient: np.ndarray
    energy: float
    half_edge_lengths: np.ndarray
    face_angles: np.ndarray
    target_curvature: np.ndarray
    initial_half_edge_lengths: np.ndarray | None = None
    landmark_indices: np.ndarray | None = None
    initial_landmark_distances: np.ndarray | None = None
    base_surface_id: SurfaceId | None = None

    @property
    def area_distortion_ratio(self) -> np.ndarray:
        """Computes the elementwise area distortion ratio exp(2u)."""
        return np.exp(2.0 * self.u)

    @property
    def max_scale_shear(self) -> float:
        """Computes the maximum scale factor difference across adjacent edge endpoints."""
        if len(self.u) == 0:
            return 0.0
        return float(np.max(self.u) - np.min(self.u))


@dataclass
class CanonicalBridgeResult:
    """Dataclass holding all output artifacts for Module 3 (Canonical Layout & Bi-Directional Bridge)."""

    placed_triangles: Dict[FaceId, Dict[Any, Any]]
    local_dilatation_map: Dict[FaceId, float]
    holonomy_seams: Dict[Any, Any]
    face_sequence: List[FaceId]
    poincare_arc_samples: np.ndarray
    grid_path_coords: np.ndarray
    is_hyperbolic: bool
