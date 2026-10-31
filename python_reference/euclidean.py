import numpy as np
from typing import Dict, Tuple, List, Any
import networkx as nx
import collections
from absl import logging

from mazedemo.viewer import Viewer, Face, Vertex, HalfEdge
from mazedemo.ids import SurfaceId, FaceId, HalfEdgeId, VertexId
from mazedemo.surface_factory import SurfaceFactory


def compute_euclidean_angles_from_opposite_lengths(
    lengths: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Computes the three angles of a Euclidean triangle given its edge lengths.
    The angles are returned in the same order as the lengths they are opposite to.
    """
    a, b, c = lengths
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Edge lengths must be strictly positive. Got {lengths}")

    # Normalize lengths to make the triangle inequality check scale-invariant
    max_len = max(a, b, c)
    a_norm, b_norm, c_norm = a / max_len, b / max_len, c / max_len

    if (a_norm + b_norm <= c_norm + 1e-9 or 
        a_norm + c_norm <= b_norm + 1e-9 or 
        b_norm + c_norm <= a_norm + 1e-9):
        if (np.isclose(a_norm + b_norm, c_norm) or 
            np.isclose(a_norm + c_norm, b_norm) or 
            np.isclose(b_norm + c_norm, a_norm)):
            raise ValueError(
                f"Degenerate triangles are not supported. Got lengths {lengths}"
            )
        else:
            raise ValueError(f"Triangle inequality violated for lengths {lengths}")

    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

    return (
        np.arccos(np.clip(cos_A, -1.0, 1.0)),
        np.arccos(np.clip(cos_B, -1.0, 1.0)),
        np.arccos(np.clip(cos_C, -1.0, 1.0)),
    )

def calculate_euclidean_flip_score(he: HalfEdge) -> float:
    r"""
    Calculates the "illegality" score for Euclidean Delaunay triangulations.
    Uses the cotan condition: score = -(cot(alpha) + cot(beta)).
    Positive score means the edge is Non-Delaunay.
    """
    try:
        # In triangle 1 (he's face), the angle alpha is opposite the edge 'he'.
        # The sides of this triangle are he, he.next, and he.prev.
        # The compute_euclidean_angles_from_opposite_lengths function returns angles
        # opposite the lengths in the order they are provided.
        lengths1 = (he.length, he.next.length, he.prev.length)
        angles1 = compute_euclidean_angles_from_opposite_lengths(lengths1)
        alpha = angles1[0]

        # In triangle 2 (he.twin's face), the angle beta is opposite the edge 'he.twin'.
        lengths2 = (he.twin.length, he.twin.next.length, he.twin.prev.length)
        angles2 = compute_euclidean_angles_from_opposite_lengths(lengths2)
        beta = angles2[0]

        # The score is positive if alpha + beta > pi (for a convex quad).
        # This is equivalent to cot(alpha) + cot(beta) < 0.
        # The score is defined as -(cot(alpha) + cot(beta)).
        # So a positive score means the edge is non-Delaunay.
        if alpha > 1e-9 and beta > 1e-9:
            cot_alpha = 1.0 / np.tan(alpha)
            cot_beta = 1.0 / np.tan(beta)
            return -(cot_alpha + cot_beta)
        else:
            # Degenerate triangle, this edge cannot be flipped.
            return -1.0

    except (AttributeError, TypeError, ValueError, IndexError):
        # If any length is missing or invalid, return a score indicating a stable edge.
        return -1.0


def calculate_euclidean_flip_length(he: HalfEdge) -> float:
    """Calculates the new diagonal length using the Euclidean Law of Cosines."""
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
        angle_bca, _, _ = compute_euclidean_angles_from_opposite_lengths(lengths_bca)

        # Triangle ACD (he.face). Its vertices are A, C, D.
        # The angle ACD is at vertex C, opposite edge AD (he.prev).
        lengths_acd = (he.prev.length, he.length, he.next.length)
        angle_acd, _, _ = compute_euclidean_angles_from_opposite_lengths(lengths_acd)

        angle_bcd = angle_bca + angle_acd

        # Sides of triangle BCD
        len_bc = he.twin.prev.length
        len_cd = he.next.length

        # Law of Cosines: BD^2 = BC^2 + CD^2 - 2*BC*CD*cos(BCD)
        new_len_sq = len_bc**2 + len_cd**2 - 2 * len_bc * len_cd * np.cos(angle_bcd)

        return np.sqrt(max(0, new_len_sq))

    except (AttributeError, TypeError, ValueError, IndexError):
        # If any length is missing or invalid (e.g., boundary edge),
        # return a value indicating failure.
        return -1.0


class EuclideanGeometry:
    """A collection of Euclidean-specific geometric primitives."""

    def get_centered_face_embedding(self, face: Face) -> Dict[VertexId, np.ndarray]:
        """
        Creates a canonical Euclidean embedding for a single face, centered at the origin.
        Returns points as 3D homogeneous coordinates [x, y, 1].
        """
        hes = face.half_edges
        if len(hes) != 3:
            raise ValueError(
                "Canonical embedding is only defined for triangular faces."
            )

        v0, v1, v2 = (he.origin for he in hes)

        len_v0v1 = hes[0].length
        len_v1v2 = hes[1].length
        len_v2v0 = hes[2].length

        assert len_v0v1 is not None and len_v1v2 is not None and len_v2v0 is not None

        # 1. Construct the triangle with one vertex at the origin.
        p0 = np.array([0.0, 0.0, 1.0])
        p1 = np.array([len_v0v1, 0.0, 1.0])

        # From Law of Cosines: c^2 = a^2 + b^2 - 2ab*cos(C)
        # Here, a=len_v2v0, b=len_v0v1, c=len_v1v2. Angle is at p0 (v0).
        # x2 is the projection of the v0-v2 edge onto the v0-v1 edge.
        if np.isclose(len_v0v1, 0):
            x2 = 0.0
        else:
            x2 = (len_v2v0**2 + len_v0v1**2 - len_v1v2**2) / (2 * len_v0v1)

        y2_sq = len_v2v0**2 - x2**2
        if y2_sq < 0:
            # Clamp to zero to handle floating point inaccuracies for degenerate triangles
            y2 = 0.0
            logging.warning(
                f"Negative y2_sq ({y2_sq}) encountered for face {face.id}. Clamping to 0."
            )
        else:
            y2 = np.sqrt(y2_sq)

        p2 = np.array([x2, y2, 1.0])

        # 2. Compute the barycenter of the constructed triangle.
        barycenter = (p0 + p1 + p2) / 3.0

        # 3. Translate all points to center the barycenter at the origin.
        translation_vec = np.array([barycenter[0], barycenter[1], 0.0])
        return {
            v0.id: p0 - translation_vec,
            v1.id: p1 - translation_vec,
            v2.id: p2 - translation_vec,
        }

    def get_isometry_aligning_edges(
        self, p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
        """Computes a 2D Euclidean transformation to map edge (p1, p2) to (q1, q2)."""
        # Check length of the 2D part of the homogeneous vectors (using scale-tolerant rtol/atol).
        if not np.isclose(np.linalg.norm(p2 - p1), np.linalg.norm(q2 - q1), rtol=1e-2, atol=1e-4):
            raise ValueError(
                f"Euclidean edge lengths must match to be aligned: {np.linalg.norm(p2 - p1)} vs {np.linalg.norm(q2 - q1)}"
            )

        T1 = np.array([[1, 0, -p1[0]], [0, 1, -p1[1]], [0, 0, 1]])
        # Direction vectors have w=0, so simple subtraction is correct.
        v_p = p2 - p1
        v_q = q2 - q1
        angle = np.arctan2(v_q[1], v_q[0]) - np.arctan2(v_p[1], v_p[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, q1[0]], [0, 1, q1[1]], [0, 0, 1]])
        return T2 @ R @ T1

    def calculate_dual_edge_weight(
        self, f1_embedding: Dict, f2_embedding: Dict, isometry: np.ndarray
    ) -> float:
        """Calculates distance between face centroids for dual graph edge weight."""
        centroid_f1 = np.mean(list(f1_embedding.values()), axis=0)
        centroid_f2_canonical = np.mean(list(f2_embedding.values()), axis=0)

        # Transform f2's centroid into f1's frame to measure distance
        centroid_f2_transformed = isometry @ centroid_f2_canonical

        # Calculate distance between the 2D parts of the homogeneous coordinates.
        return np.linalg.norm(centroid_f1[:2] - centroid_f2_transformed[:2])

    def identity_transform(self) -> np.ndarray:
        """Returns the 3x3 identity matrix for 2D homogeneous coordinates."""
        return np.identity(3)

    def invert_transform(self, transform: np.ndarray) -> np.ndarray:
        """Returns the inverse of a Euclidean transformation matrix."""
        return np.linalg.inv(transform)

    def is_isometry(self, transform: np.ndarray, tol=1e-9) -> bool:
        """Checks if a 3x3 matrix is a valid 2D Euclidean isometry."""
        if transform.shape != (3, 3):
            return False

        # Check bottom row is [0, 0, 1]
        if not np.allclose(transform[2, :], [0, 0, 1], atol=tol):
            return False

        # Check rotation part is orthogonal and orientation-preserving
        R = transform[:2, :2]
        if not np.allclose(R.T @ R, np.identity(2), atol=tol):
            return False
        if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
            return False

        return True

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculates the Euclidean distance between the 2D components of two points."""
        return np.linalg.norm(p1[:2] - p2[:2])

    def distance_sq(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculates the squared Euclidean distance between the 2D components of two points."""
        return np.sum((p1[:2] - p2[:2]) ** 2)

    def compute_angles_from_opposite_lengths(
        self, lengths: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Computes the three angles of a Euclidean triangle given its edge lengths."""
        return compute_euclidean_angles_from_opposite_lengths(lengths)

    def compute_length_from_corner(
        self, angle: float, len1: float, len2: float
    ) -> float:
        """
        Computes the length of the side opposite a corner in a Euclidean triangle,
        given the angle at that corner and the lengths of the two adjacent sides.
        Uses the Law of Cosines: c^2 = a^2 + b^2 - 2ab*cos(C).
        """
        len_sq = len1**2 + len2**2 - 2 * len1 * len2 * np.cos(angle)
        return np.sqrt(max(0, len_sq))
