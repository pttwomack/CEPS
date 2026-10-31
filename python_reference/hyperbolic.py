# mazedemo/hyperbolic.py
"""
A toolkit for hyperbolic geometry calculations using the hyperboloid model.

This module provides functions for working with points, distances, and isometries
in 2D hyperbolic space, represented on the forward sheet of a hyperboloid in 3D
Minkowski space.

The hyperboloid model is chosen for its numerical stability and the linear
representation of isometries as 3x3 Lorentz transformation matrices.

Terminology:
- A point in H^2 is represented as a 3D numpy array (x, y, z) such that
  x^2 + y^2 - z^2 = -1 and z > 0.
- The Lorentz inner product of u and v is <u, v>_L = u[0]v[0] + u[1]v[1] - u[2]v[2].
- The distance between two points u, v is arccosh(-<u, v>_L).
- Isometries of H^2 are represented by 3x3 matrices that preserve the Lorentz
  inner product (Lorentz transformations).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List
import numpy as np
from absl import logging
from scipy.linalg import expm
import scipy.special
from collections import namedtuple

from mazedemo.ids import VertexId

if TYPE_CHECKING:
    from mazedemo.viewer import Face, HalfEdge

# --- Low-level primitive functions for hyperbolic geometry ---


def lorentz_dot(u, v):
    """
    Computes the Lorentz inner product of two vectors in R^3.
    <u, v>_L = u[0]v[0] + u[1]v[1] - u[2]v[2]
    """
    return u[0] * v[0] + u[1] * v[1] - u[2] * v[2]


def is_on_hyperboloid(p, tol=1e-9):
    """
    Checks if a point p lies on the forward sheet of the hyperboloid.
    """
    return np.isclose(lorentz_dot(p, p), -1.0, atol=tol) and p[2] > 0


def get_lorentz_boost(v, t):
    """
    Creates a Lorentz boost (hyperbolic translation) matrix.
    This moves points along the geodesic defined by the direction vector v.

    Args:
        v (np.ndarray): A 2D direction vector [vx, vy].
        t (float): The distance to translate.

    Returns:
        np.ndarray: A 3x3 Lorentz transformation matrix.
    """
    vx, vy = v
    norm_sq = vx**2 + vy**2
    if not np.isclose(norm_sq, 1.0):
        norm = np.sqrt(norm_sq)
        if norm > 0:
            vx /= norm
            vy /= norm

    # Generator of the boost
    K = np.array([[0, 0, vx], [0, 0, vy], [vx, vy, 0]])
    return expm(t * K)


def distance(p1, p2):
    """
    Computes the hyperbolic distance between two points on the hyperboloid.
    """
    dot_product = lorentz_dot(p1, p2)
    # Clamp the dot product to handle potential floating point errors
    # The argument to arccosh must be >= 1. Since dot_product is <= -1, -dot_product >= 1.
    arg = -np.clip(dot_product, -np.inf, -1.0)
    return np.arccosh(arg)


def get_transform_to_canonical(p1, p2):
    """
    Computes the transformation that maps an edge (p1, p2) to a canonical
    position, where p1 is at the origin (0,0,1) and p2 is in the x-z plane.
    """
    # 1. Boost p1 to the origin.
    dist_p1_origin = distance(np.array([0.0, 0.0, 1.0]), p1)
    if np.isclose(dist_p1_origin, 0):
        boost_to_origin = np.identity(3)
    else:
        dir_p1 = p1[:2]
        # The boost to move p1 to origin is a boost in the opposite direction.
        boost_to_origin = get_lorentz_boost(dir_p1, -dist_p1_origin)

    # 2. Rotate the boosted p2' into the x-z plane.
    p2_prime = boost_to_origin @ p2
    px, py = p2_prime[:2]
    norm = np.sqrt(px**2 + py**2)

    if np.isclose(norm, 0):
        # p2' is already at the origin, no rotation needed.
        rotation = np.identity(3)
    else:
        cos_a = px / norm
        sin_a = py / norm
        # Rotation matrix for angle -a to align with the x-axis.
        rotation = np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]])

    return rotation @ boost_to_origin


def get_isometry_aligning_edges(p1, p2, q1, q2):
    """
    Computes the Lorentz transformation that maps edge (p1, p2) to (q1, q2).
    This is the core "gluing" operation for constructing the domain.
    The logic is: M = T_canonical_to_q @ T_p_to_canonical
    """
    # Check that edge lengths are the same before proceeding (using scale-tolerant rtol/atol).
    if not np.isclose(distance(p1, p2), distance(q1, q2), rtol=1e-2, atol=1e-4):
        raise ValueError(
            f"Edges lengths {distance(p1, p2)}, {distance(q1, q2)} must match to be aligned."
        )

    T_p_to_canonical = get_transform_to_canonical(p1, p2)
    T_q_to_canonical = get_transform_to_canonical(q1, q2)

    # The transform from canonical space to q-space is the inverse of T_q_to_canonical.
    # For a Lorentz transform M, inv(M) = J @ M.T @ J, where J = diag(1,1,-1).
    J = np.diag([1.0, 1.0, -1.0])
    T_canonical_to_q = J @ T_q_to_canonical.T @ J

    isometry = T_canonical_to_q @ T_p_to_canonical

    # Compose the transforms: first map p to canonical, then canonical to q.
    logging.vlog(9, "local isometry should match %s->%s, %s->%s", p1, q1, p2, q2)

    return isometry


def construct_triangle(len_a, len_b, len_c):
    """
    Constructs a hyperbolic triangle in the hyperboloid model from its side lengths.

    The triangle is constructed in a standard position:
    - Vertex A is at the "origin" (0, 0, 1).
    - Vertex B is placed along the x-z plane.
    - Vertex C is placed in the x-y-z space.

    Args:
        len_a (float): The length of the side opposite vertex A (side BC).
        len_b (float): The length of the side opposite vertex B (side AC).
        len_c (float): The length of the side opposite vertex C (side AB).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        3D coordinates of the three vertices (A, B, C).
    """
    # Validate triangle inequality
    if not (len_a + len_b > len_c and len_a + len_c > len_b and len_b + len_c > len_a):
        tol = 1e-9
        if not (
            len_a + len_b > len_c - tol
            and len_a + len_c > len_b - tol
            and len_b + len_c > len_a - tol
        ):
            raise ValueError(
                f"Provided lengths do not satisfy the triangle inequality. {len_a}, {len_b}, `{len_c}"
            )
        else:
            if len_a + len_b >= len_c:
                len_c -= tol
            elif len_a + len_c >= len_b:
                len_b -= tol
            elif len_b + len_c >= len_a:
                len_a -= tol

    # Vertex A is at the origin of the hyperboloid
    v_a = np.array([0.0, 0.0, 1.0])

    # Place Vertex B at distance len_c from A along the x-axis direction
    boost_c = get_lorentz_boost(np.array([1.0, 0.0]), len_c)
    v_b = boost_c @ v_a

    # Determine Vertex C using the other two side lengths.
    cosh_b = np.cosh(len_b)
    cosh_a = np.cosh(len_a)
    z_c = cosh_b

    sinh_c = np.sinh(len_c)
    if np.isclose(sinh_c, 0):
        x_c = np.sinh(len_b)
    else:
        x_c = (np.cosh(len_c) * z_c - cosh_a) / sinh_c

    y_c_sq = z_c**2 - x_c**2 - 1
    if y_c_sq < 0:
        y_c_sq = 0

    y_c = np.sqrt(y_c_sq)
    v_c = np.array([x_c, y_c, z_c])

    # Sanity checks
    if not (
        is_on_hyperboloid(v_a) and is_on_hyperboloid(v_b) and is_on_hyperboloid(v_c)
    ):
        raise RuntimeError("Constructed vertices are not on the hyperboloid.")

    if not (
        np.isclose(distance(v_a, v_b), len_c)
        and np.isclose(distance(v_a, v_c), len_b)
        and np.isclose(distance(v_b, v_c), len_a)
    ):
        raise RuntimeError(
            f"Constructed triangle for ({len_a}, {len_b}, {len_c}) has incorrect side lengths."
        )

    return v_a, v_b, v_c


def compute_hyperbolic_angles_from_opposite_lengths(lengths):
    """
    Computes the angles of a hyperbolic triangle given its side lengths
    using the hyperbolic law of cosines. Handles degenerate triangles.
    """
    a, b, c = lengths
    if any(l is None or not np.isfinite(l) or l <= 0 for l in lengths):
        raise ValueError(
            f"Invalid lengths provided for hyperbolic angle calculation: {lengths}"
        )

    if a + b <= c + 1e-9 or a + c <= b + 1e-9 or b + c <= a + 1e-9:
        raise ValueError(f"Triangle inequality violated for lengths: {lengths}")

    cosh_A = np.clip(
        (np.cosh(b) * np.cosh(c) - np.cosh(a)) / (np.sinh(b) * np.sinh(c)), -1.0, 1.0
    )
    cosh_B = np.clip(
        (np.cosh(a) * np.cosh(c) - np.cosh(b)) / (np.sinh(a) * np.sinh(c)), -1.0, 1.0
    )
    cosh_C = np.clip(
        (np.cosh(a) * np.cosh(b) - np.cosh(c)) / (np.sinh(a) * np.sinh(b)), -1.0, 1.0
    )

    return (np.arccos(cosh_A), np.arccos(cosh_B), np.arccos(cosh_C))


def lobachevsky(x: np.ndarray) -> np.ndarray:
    """
    Computes the Lobachevsky function Λ(x) = 0.5 * Im(Li_2(e^{2ix})).
    This uses the identity Li_2(z) = spence(1-z), where spence is the
    scipy library function for the dilogarithm.
    """
    x = np.asarray(x)
    return 0.5 * np.imag(scipy.special.spence(1.0 - np.exp(2j * x)))


def volume_of_ideal_tetrahedron(dihedral_angles: tuple[float, float, float]) -> float:
    """
    Computes the volume of an ideal hyperbolic tetrahedron, which is given by
    the sum of Lobachevsky function values of its dihedral angles.
    """
    return np.sum(lobachevsky(np.asarray(dihedral_angles)))


def hyperboloid_to_poincare(p: np.ndarray) -> np.ndarray:
    """
    Projects a point from the hyperboloid model to the Poincaré disk model.
    """
    return p[:2] / (1 + p[2])


# --- Barycenter Implementations (moved from barycenter.py) ---


def einstein_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Performs Einstein velocity addition in the hyperboloid model.

    This operation is the fundamental addition in gyrovector space and is
    equivalent to composing the Lorentz boosts that map the origin to u and v.
    """
    # This is a numerically stable and geometrically correct implementation.
    # The sum of two boosts is not simply additive. However, the midpoint M
    # of two points u and v on the hyperboloid is proportional to their
    # Euclidean sum. We can find it by adding the vectors and re-projecting
    # to the hyperboloid. This is a special case of Einstein addition.
    midpoint_vector = u + v
    return midpoint_vector / np.sqrt(-lorentz_dot(midpoint_vector, midpoint_vector))


def simple_projection_barycenter(vert_positions: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the barycenter of a hyperbolic triangle using a simple
    Euclidean average followed by a projection onto the hyperboloid.

    NOTE: This method is fast but not idempotent. It is suitable for
    applications like visualization where performance is critical and

    perfect canonicity is not required.
    """
    if len(vert_positions) != 3:
        raise ValueError("Input must be a list of three vertex positions.")

    # Simple Euclidean average of the 3D vertex coordinates.
    p_e = sum(vert_positions) / 3.0

    # Project the point back onto the hyperboloid.
    return p_e / np.sqrt(-lorentz_dot(p_e, p_e))


def iterative_projection_barycenter(vert_positions: List[np.ndarray]) -> np.ndarray:
    """
    NOTE: This method is deliberately not implemented and is considered an
    antipattern for this project.
    """
    raise NotImplementedError(
        "This iterative method is an antipattern; use gyro_barycenter instead."
    )


def gyro_barycenter(vert_positions: List[np.ndarray]) -> np.ndarray:
    """
    Computes the canonical barycenter (gyrocentroid) of a hyperbolic triangle.

    This method is idempotent and represents the true geometric center of the
    triangle. It is the preferred method for all canonical geometric
    calculations. This implementation finds the centroid as the intersection
    of the triangle's medians.
    """
    if len(vert_positions) != 3:
        raise ValueError("Input must be a list of three vertex positions.")

    a, b, c = vert_positions

    # The "gyromidpoint" of a segment (e.g., BC) is found using Einstein addition.
    m_a = einstein_add(b, c)
    m_b = einstein_add(a, c)

    # The centroid is the intersection of the two medians (A to m_a and B to m_b).
    n1 = np.cross(a, m_a)
    n2 = np.cross(b, m_b)

    line_direction = np.cross(n1, n2)
    centroid = line_direction / np.sqrt(-lorentz_dot(line_direction, line_direction))

    # Ensure we have the point on the forward sheet (z > 0).
    if centroid[2] < 0:
        centroid *= -1

    return centroid


def volume_of_hyperideal_tetrahedron(theta: List[float], alpha: List[float]) -> float:
    """
    Computes the volume of a hyperbolic tetrahedron with one hyperideal vertex
    and three ideal vertices.

    Args:
        theta: 3x angles of apex triangle
        alpha: 3x dihedral angles at opposite edges

    Returns:
        volume of hyperideal tetrahedron.
    """
    # See: [Ushijima_2006]
    a = np.exp(1j * theta[0])
    b = np.exp(1j * theta[1])
    c = np.exp(1j * theta[2])
    d = np.exp(1j * alpha[0])
    e = np.exp(1j * alpha[1])
    f = np.exp(1j * alpha[2])

    p = a * b * c * d * e * f

    # Coefficients (Derived from Ushijima 2006, Murakami-Yano)
    Alpha = (
        1.0
        + (a * b * d * e)
        + (a * c * d * f)
        + (b * c * e * f)
        + (a * b * c)
        + (a * e * f)
        + (b * d * f)
        + (c * d * e)
    )

    # Beta coefficient (Linear term)
    term1 = (a - 1.0 / a) * (d - 1.0 / d)
    term2 = (b - 1.0 / b) * (e - 1.0 / e)
    term3 = (c - 1.0 / c) * (f - 1.0 / f)
    Beta = -p * (term1 + term2 + term3)

    # Gamma coefficient (Constant term)
    Gamma = p**2 * np.conj(Alpha)

    # 3. Solve Quadratic: Alpha * z^2 + Beta * z + Gamma = 0
    discriminant = np.sqrt(Beta**2 - 4 * Alpha * Gamma)
    z1 = (-Beta - discriminant) / (2 * Alpha)
    z2 = (-Beta + discriminant) / (2 * Alpha)

    def li2(z):
        return scipy.special.spence(1.0 - z)

    def U(z):
        # Ushijima Eq 1.1 definition of function U
        terms = [
            li2(z),
            li2(z * a * b * d * e),
            li2(z * a * c * d * f),
            li2(z * b * c * e * f),
            -li2(-z * a * b * c),
            -li2(-z * a * e * f),
            -li2(-z * b * d * f),
            -li2(-z * c * d * e),
        ]
        return 0.5 * np.sum(terms)

    # 5. Volume Formula
    vol = 0.5 * np.imag(U(z1) - U(z2))
    return np.abs(vol)


class HyperbolicGeometry:
    """Hyperbolic implementations of a collection of geometric primitives"""

    def get_centered_face_embedding(self, face: Face) -> Dict[VertexId, np.ndarray]:
        """
        Creates a canonical hyperbolic embedding for a single face with
        its barycenter at the origin.
        """
        hes = face.half_edges
        if len(hes) != 3:
            raise ValueError(
                "Canonical embedding is only defined for triangular faces."
            )

        v0, v1, v2 = (he.origin for he in hes)

        len_opp_v0 = hes[1].length
        len_opp_v1 = hes[2].length
        len_opp_v2 = hes[0].length
        assert len_opp_v0 and len_opp_v1 and len_opp_v2
        p0, p1, p2 = construct_triangle(len_opp_v0, len_opp_v1, len_opp_v2)

        barycenter = gyro_barycenter([p0, p1, p2])
        transform_to_origin = get_transform_to_canonical(barycenter, p0)

        return {
            v0.id: transform_to_origin @ p0,
            v1.id: transform_to_origin @ p1,
            v2.id: transform_to_origin @ p2,
        }

    def get_isometry_aligning_edges(
        self, p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
        """Computes a 3D hyperbolic transformation to map edge (p1, p2) to (q1, q2)."""
        return get_isometry_aligning_edges(p1, p2, q1, q2)

    def calculate_dual_edge_weight(
        self, f1_embedding: Dict, f2_embedding: Dict, isometry: np.ndarray
    ) -> float:
        """Calculates distance between face centroids for dual graph edge weight."""
        centroid_f1 = gyro_barycenter(list(f1_embedding.values()))
        centroid_f2_canonical = gyro_barycenter(list(f2_embedding.values()))

        # Transform f2's centroid into f1's frame to measure distance
        centroid_f2_transformed = (isometry @ centroid_f2_canonical.T).T

        return distance(centroid_f1, centroid_f2_transformed)

    def identity_transform(self) -> np.ndarray:
        """Returns the 3x3 identity matrix."""
        return np.identity(3)

    def invert_transform(self, transform: np.ndarray) -> np.ndarray:
        """Returns the inverse of a Lorentz transformation matrix."""
        # For a Lorentz transform M, inv(M) = J @ M.T @ J, where J = diag(1,1,-1).
        J = np.diag([1.0, 1.0, -1.0])
        return J @ transform.T @ J

    def is_isometry(self, transform: np.ndarray, tol=1e-9) -> bool:
        """Checks if a 3x3 matrix is a valid hyperbolic isometry (Lorentz transform)."""
        if transform.shape != (3, 3):
            return False

        # Check that it preserves the Lorentz inner product: M.T @ J @ M = J
        J = np.diag([1.0, 1.0, -1.0])
        if not np.allclose(transform.T @ J @ transform, J, atol=tol):
            return False

        # Check that it's orientation-preserving
        if not np.isclose(np.linalg.det(transform), 1.0, atol=tol):
            return False

        # Check that it maps the forward sheet to itself (M_22 >= 1)
        if transform[2, 2] < 1.0 - tol:
            return False

        return True

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Computes the hyperbolic distance between two points on the hyperboloid."""
        return distance(p1, p2)

    def compute_angles_from_opposite_lengths(
        self, lengths: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Computes the angles of a hyperbolic triangle given its side lengths."""
        return compute_hyperbolic_angles_from_opposite_lengths(lengths)

    def compute_length_from_corner(
        self, angle: float, len1: float, len2: float
    ) -> float:
        """
        Computes the length of the side opposite a corner in a hyperbolic triangle,
        given the angle at that corner and the lengths of the two adjacent sides.
        Uses the Hyperbolic Law of Cosines:
        cosh(c) = cosh(a)cosh(b) - sinh(a)sinh(b)cos(C).
        """
        cosh_c = np.cosh(len1) * np.cosh(len2) - np.sinh(len1) * np.sinh(len2) * np.cos(
            angle
        )
        # Argument to arccosh must be >= 1. Clamp to handle floating point errors.
        return np.arccosh(max(1.0, cosh_c))


def calculate_hyperbolic_flip_score(he: HalfEdge) -> float:
    """
    Calculates a score to determine if an edge should be flipped in hyperbolic geometry.
    Positive score means flip. Based on the Delaunay condition.
    An edge is Delaunay iff sum of opposite angles <= pi.
    """
    if he.twin is None:
        return -1.0  # Boundary edges are always Delaunay

    # Face 1 (of he)
    # The angle opposite 'he' is at the vertex he.prev.origin
    hes1 = (he, he.next, he.prev)
    lengths1 = (hes1[1].length, hes1[2].length, hes1[0].length)
    angles1 = compute_hyperbolic_angles_from_opposite_lengths(lengths1)
    angle_in_face1 = angles1[2]

    # Face 2 (of he.twin)
    # The angle opposite 'he.twin' is at the vertex he.twin.prev.origin
    hes2 = (he.twin, he.twin.next, he.twin.prev)
    lengths2 = (hes2[1].length, hes2[2].length, hes2[0].length)
    angles2 = compute_hyperbolic_angles_from_opposite_lengths(lengths2)
    angle_in_face2 = angles2[2]

    return angle_in_face1 + angle_in_face2 - np.pi


def calculate_hyperbolic_flip_length(he: HalfEdge) -> float:
    """
    Calculates the length of a flipped edge in hyperbolic geometry using
    the hyperbolic Ptolemy's theorem for an ideal cyclic quadrilateral.
    sinh(p/2)sinh(q/2) = sinh(a/2)sinh(c/2) + sinh(b/2)sinh(d/2)
    """
    # p is the length of the diagonal to be flipped.
    # a, b, c, d are the lengths of the four outer edges of the quadrilateral.
    p = he.length
    a = he.prev.length
    b = he.next.length
    c = he.twin.prev.length
    d = he.twin.next.length

    if any(l is None for l in [p, a, b, c, d]):
        raise ValueError(
            "Cannot calculate hyperbolic flip length with missing edge lengths."
        )

    # Avoid division by zero for degenerate edges.
    if np.sinh(p / 2.0) <= 1e-9:
        raise ValueError("Degenerate edge in hyperbolic flip length calculation")

    # Hyperbolic Ptolemy's theorem solved for sinh(q/2), where q is the new diagonal length.
    term1 = np.sinh(a / 2.0) * np.sinh(c / 2.0)
    term2 = np.sinh(b / 2.0) * np.sinh(d / 2.0)

    sinh_q_half = (term1 + term2) / np.sinh(p / 2.0)

    # arcsinh can handle any real input, so this is safe.
    q = 2 * np.arcsinh(sinh_q_half)
    return q
