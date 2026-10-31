import numpy as np
from typing import Dict, Tuple, List
from absl import logging

from mazedemo.ids import VertexId
from mazedemo.viewer import HalfEdge, Face


class SphericalGeometry:
    """Spherical implementations of a collection of geometric primitives."""

    def compute_edge_lengths(
        self, log_radii: np.ndarray, cos_phi_map: Dict[frozenset, float]
    ) -> Dict[frozenset, float]:
        """
        Calculates spherical edge lengths from log-radii and fixed cos(phi_ij) invariants.
        Formula: cos(l_ij) = cos(r_i)cos(r_j) - sin(r_i)sin(r_j)cos(φ_ij)
        """
        lengths = {}
        for edge, cos_phi_ij in cos_phi_map.items():
            v_i, v_j = tuple(edge)
            r_i = np.exp(log_radii[v_i])
            r_j = np.exp(log_radii[v_j])

            cos_l_ij = (
                np.cos(r_i) * np.cos(r_j) - np.sin(r_i) * np.sin(r_j) * cos_phi_ij
            )
            # Clamp to avoid numerical issues outside [-1, 1]
            cos_l_ij = np.clip(cos_l_ij, -1.0, 1.0)
            lengths[edge] = np.arccos(cos_l_ij)
        return lengths

    def compute_angles_from_opposite_lengths(
        self, lengths: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Computes the three angles of a spherical triangle given its edge lengths
        using the spherical law of cosines.
        """
        a, b, c = lengths
        if any(l is None or not np.isfinite(l) or l <= 0 for l in lengths):
            # Degenerate triangle, return zero angles
            return (0.0, 0.0, 0.0)

        # Spherical Law of Cosines: cos(A) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c))
        # Clamp arguments to arccos to avoid numerical issues
        cos_A = (np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c))
        cos_B = (np.cos(b) - np.cos(a) * np.cos(c)) / (np.sin(a) * np.sin(c))
        cos_C = (np.cos(c) - np.cos(a) * np.cos(b)) / (np.sin(a) * np.sin(b))

        return (
            np.arccos(np.clip(cos_A, -1.0, 1.0)),
            np.arccos(np.clip(cos_B, -1.0, 1.0)),
            np.arccos(np.clip(cos_C, -1.0, 1.0)),
        )

    def compute_flipped_edge_length(self, he: HalfEdge) -> float:
        """
        Calculates the new diagonal length using the Spherical Law of Cosines.
        This is analogous to calculate_euclidean_flip_length but for spherical geometry.
        """
        try:
            # The quadrilateral is formed by the vertices:
            # A = he.origin
            # B = he.twin.prev.origin
            # C = he.next.origin
            # D = he.prev.origin
            # The original diagonal is A-C (he). The new diagonal is B-D.

            # We will compute the length of B-D using the Law of Cosines on triangle BCD.
            # For this, we need lengths of BC, CD and the angle BCD.
            # Angle BCD = Angle BCA + Angle ACD.

            # Triangle BCA (he.twin.face). Its vertices are B, C, A.
            # The angle BCA is at vertex C, opposite edge AB (he.twin.next).
            lengths_bca = (he.twin.next.length, he.twin.prev.length, he.twin.length)
            angle_bca, _, _ = self.compute_angles_from_opposite_lengths(lengths_bca)

            # Triangle ACD (he.face). Its vertices are A, C, D.
            # The angle ACD is at vertex C, opposite edge AD (he.prev).
            lengths_acd = (he.prev.length, he.length, he.next.length)
            angle_acd, _, _ = self.compute_angles_from_opposite_lengths(lengths_acd)

            angle_bcd = angle_bca + angle_acd

            # Sides of triangle BCD
            len_bc = he.twin.prev.length
            len_cd = he.next.length

            # Spherical Law of Cosines: cos(BD) = cos(BC)cos(CD) + sin(BC)sin(CD)cos(BCD)
            cos_new_len = np.cos(len_bc) * np.cos(len_cd) + np.sin(len_bc) * np.sin(
                len_cd
            ) * np.cos(angle_bcd)
            return np.arccos(np.clip(cos_new_len, -1.0, 1.0))

        except (AttributeError, TypeError, ValueError, IndexError):
            logging.warning(
                "Failed to compute spherical flip length for half-edge %s", he.id
            )
            return -1.0  # Indicate failure

    def compute_flipped_cos_phi(
        self, new_length: float, r_k: float, r_l: float
    ) -> float:
        """
        Inverts the spherical length formula to find cos(phi_kl) for a new edge.
        cos(φ_kl) = (cos(l_kl) - cos(r_k)cos(r_l)) / (-sin(r_k)sin(r_l))
        """
        sin_rk_sin_rl = np.sin(r_k) * np.sin(r_l)
        if np.isclose(sin_rk_sin_rl, 0):
            return 0.0  # Avoid division by zero, effectively a degenerate case

        cos_phi_kl = (np.cos(new_length) - np.cos(r_k) * np.cos(r_l)) / (-sin_rk_sin_rl)
        return np.clip(cos_phi_kl, -1.0, 1.0)

    def get_centered_face_embedding(self, face: Face) -> Dict[VertexId, np.ndarray]:
        """
        Creates a canonical spherical embedding for a single face, centered at the origin.
        This is a placeholder and might need a more robust implementation for visualization.
        """
        # For now, return a dummy embedding or raise NotImplementedError
        # A proper spherical embedding would involve placing vertices on a sphere.
        # This method is primarily used for dual graph construction, which might
        # not be directly applicable in the same way for spherical geometry.
        logging.warning(
            "SphericalGeometry.get_centered_face_embedding is a placeholder."
        )
        hes = face.half_edges
        v0, v1, v2 = (he.origin for he in hes)
        return {
            v0.id: np.array([1.0, 0.0, 0.0]),
            v1.id: np.array([0.0, 1.0, 0.0]),
            v2.id: np.array([0.0, 0.0, 1.0]),
        }

    def get_isometry_aligning_edges(
        self, p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
        """
        Computes a 3D spherical transformation (rotation) to map edge (p1, p2) to (q1, q2).
        This is a placeholder.
        """
        logging.warning(
            "SphericalGeometry.get_isometry_aligning_edges is a placeholder."
        )
        return np.identity(3)

    def calculate_dual_edge_weight(
        self, f1_embedding: Dict, f2_embedding: Dict, isometry: np.ndarray
    ) -> float:
        """
        Calculates distance between face centroids for dual graph edge weight.
        This is a placeholder.
        """
        logging.warning(
            "SphericalGeometry.calculate_dual_edge_weight is a placeholder."
        )
        return 0.0

    def identity_transform(self) -> np.ndarray:
        """Returns the 3x3 identity matrix."""
        return np.identity(3)

    def invert_transform(self, transform: np.ndarray) -> np.ndarray:
        """Returns the inverse of a spherical transformation (rotation) matrix."""
        return transform.T  # For rotations, inverse is transpose

    def is_isometry(self, transform: np.ndarray, tol=1e-9) -> bool:
        """Checks if a 3x3 matrix is a valid spherical isometry (rotation)."""
        if transform.shape != (3, 3):
            return False
        # Check if it's an orthogonal matrix (R.T @ R = I)
        if not np.allclose(transform.T @ transform, np.identity(3), atol=tol):
            return False
        # Check if it's orientation-preserving (det(R) = 1)
        if not np.isclose(np.linalg.det(transform), 1.0, atol=tol):
            return False
        return True

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculates the spherical distance between two points on the unit sphere."""
        # For points on a unit sphere, distance is arccos(p1 . p2)
        dot_product = np.dot(p1, p2)
        return np.arccos(np.clip(dot_product, -1.0, 1.0))
