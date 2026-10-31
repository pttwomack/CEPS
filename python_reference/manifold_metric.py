import numpy as np
import heapq
import collections
from typing import Dict, Tuple, List, Any
from absl import logging
from mazedemo.symbol_group import reduce as reduce_holonomy_string


def _distance_point_to_segment_sq(
    p: np.ndarray, v: np.ndarray, w: np.ndarray, geometry: Any
) -> float:
    """
    Calculates the squared Euclidean distance from a point 'p' to the closest point
    on a line segment defined by 'v' and 'w'.
    Assumes points are 3D homogeneous coordinates [x, y, w] where w=1 for Euclidean.
    Relies on geometry.distance_sq for final distance calculation.
    """
    p_2d = p[:2]
    v_2d = v[:2]
    w_2d = w[:2]

    segment_vec = w_2d - v_2d
    point_vec = p_2d - v_2d

    segment_len_sq = np.dot(segment_vec, segment_vec)

    if segment_len_sq == 0.0:  # v and w are the same point
        return geometry.distance_sq(p, v)

    # Project point_vec onto segment_vec
    t = np.dot(point_vec, segment_vec) / segment_len_sq

    if t < 0.0:
        # Closest point is v
        return geometry.distance_sq(p, v)
    elif t > 1.0:
        # Closest point is w
        return geometry.distance_sq(p, w)
    else:
        # Closest point is on the segment
        projection_2d = v_2d + t * segment_vec
        projection_homogeneous = np.array([projection_2d[0], projection_2d[1], 1.0])
        return geometry.distance_sq(p, projection_homogeneous)


def _calculate_distance_to_transformed_polygon_sq(
    p: np.ndarray,
    transform_matrix: np.ndarray,
    fundamental_domain_boundary_points: List[np.ndarray],
    geometry: Any,
) -> float:
    """
    Calculates the squared Euclidean distance from a point 'p' to the closest point
    on the boundary polygon of the fundamental domain, after the polygon has been
    transformed by 'transform_matrix'.

    Args:
        p: The reference point (3D homogeneous coordinates).
        transform_matrix: The 3x3 transformation matrix.
        fundamental_domain_boundary_points: A list of 3D homogeneous coordinates
                                            defining the vertices of the fundamental domain's boundary polygon.
                                            Assumed to be ordered to form a closed polygon.
        geometry: An object providing geometry-specific methods like `distance_sq`.

    Returns:
        The minimum squared Euclidean distance from 'p' to any segment of the transformed polygon.
    """
    min_dist_sq = float("inf")

    transformed_boundary_points = [
        transform_matrix @ pt for pt in fundamental_domain_boundary_points
    ]

    # Iterate over segments of the transformed polygon
    for i in range(len(transformed_boundary_points)):
        v = transformed_boundary_points[i]
        w = transformed_boundary_points[
            (i + 1) % len(transformed_boundary_points)
        ]  # Wrap around for the last segment

        current_dist_sq = _distance_point_to_segment_sq(p, v, w, geometry)
        min_dist_sq = min(min_dist_sq, current_dist_sq)

    return min_dist_sq


def calculate_manifold_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    holonomy_transforms: Dict[str, np.ndarray],
    geometry: Any,
    fundamental_domain_boundary_points: List[np.ndarray],
    max_iterations: int = 10000,  # Safety break for infinite loops in complex cases
) -> Tuple[float, np.ndarray]:
    """
    Calculates the shortest manifold distance between two points p1 and p2
    on a surface, considering its fundamental domain and holonomy transformations.

    Args:
        p1: The first point in the root tile (3D homogeneous coordinates).
        p2: The second point in the root tile (3D homogeneous coordinates).
        holonomy_transforms: A dictionary mapping holonomy generator strings (e.g., 'a', 'A')
                             to their corresponding 3x3 transformation matrices.
        geometry: An object providing geometry-specific methods like `distance_sq`.
        fundamental_domain_boundary_points: A list of 3D homogeneous coordinates
                                            defining the vertices of the fundamental domain's boundary polygon.
                                            Assumed to be ordered to form a closed polygon.
        max_iterations: Maximum number of iterations for the priority queue search.

    Returns:
        A tuple (shortest_distance, best_p2_image).
        shortest_distance: The shortest manifold distance between p1 and an image of p2.
        best_p2_image: The image of p2 that is closest to p1, with its coordinates
                       expressed in the root tile's coordinate system.
    """

    # Initialize with the direct distance in the root tile
    min_dist_sq = geometry.distance_sq(p1, p2)
    best_transform = geometry.identity_transform()

    # Priority queue stores (min_dist_to_tile_sq, holonomy_str, transform_matrix)
    pq = []
    # TODO(P2): For Euclidean geometry, holonomy symbols commute.
    #  Consider using a canonical representation (e.g., tuple of integer coefficients for generators)
    #  for `visited_holonomies` to avoid redundant exploration of the same tile via different paths.
    visited_holonomies = {""}  # Start with the identity holonomy visited

    # Seed the search with base generators and their inverses
    base_generators = sorted([g for g in holonomy_transforms.keys() if g.islower() and len(g) == 1])

    for gen_char in base_generators:
        # Generator 'a'
        transform_g = holonomy_transforms[gen_char]
        dist_to_tile_sq_g = _calculate_distance_to_transformed_polygon_sq(
            p1, transform_g, fundamental_domain_boundary_points, geometry
        )
        heapq.heappush(pq, (dist_to_tile_sq_g, gen_char, transform_g))
        visited_holonomies.add(gen_char)

        # Inverse generator 'A'
        gen_char_inv = gen_char.upper()
        transform_g_inv = holonomy_transforms[gen_char_inv]
        dist_to_tile_sq_g_inv = _calculate_distance_to_transformed_polygon_sq(
            p1, transform_g_inv, fundamental_domain_boundary_points, geometry
        )
        heapq.heappush(pq, (dist_to_tile_sq_g_inv, gen_char_inv, transform_g_inv))
        visited_holonomies.add(gen_char_inv)

    iterations = 0
    while pq and iterations < max_iterations:
        iterations += 1

        dist_to_tile_sq, holonomy_str, current_transform = heapq.heappop(pq)

        # Pruning condition: if the closest possible point in this tile is
        # further than our current best distance, we can stop.
        if dist_to_tile_sq >= min_dist_sq:
            break

        # Calculate the image of p2 in this tile and update min_dist_sq if closer
        p2_image_in_current_tile = current_transform @ p2
        current_dist_sq = geometry.distance_sq(p1, p2_image_in_current_tile)

        if current_dist_sq < min_dist_sq:
            min_dist_sq = current_dist_sq
            best_transform = current_transform

        # Expand the wavefront
        for gen_char_next in base_generators:
            # Expand by both the generator and its inverse
            gen_char_inv_next = gen_char_next.upper()

            next_transform_matrix = holonomy_transforms[gen_char_next]
            new_transform = current_transform @ next_transform_matrix
            new_holonomy_str = holonomy_str + gen_char_next

            reduced_holonomy_str = reduce_holonomy_string(new_holonomy_str)

            if reduced_holonomy_str not in visited_holonomies:
                visited_holonomies.add(reduced_holonomy_str)

                new_dist_to_tile_sq = _calculate_distance_to_transformed_polygon_sq(
                    p1, new_transform, fundamental_domain_boundary_points, geometry
                )
                heapq.heappush(
                    pq, (new_dist_to_tile_sq, reduced_holonomy_str, new_transform)
                )

            next_transform_matrix_inv = holonomy_transforms[gen_char_inv_next]
            new_transform_inv = current_transform @ next_transform_matrix_inv
            new_holonomy_str_inv = holonomy_str + gen_char_inv_next

            reduced_holonomy_str_inv = reduce_holonomy_string(new_holonomy_str_inv)

            if reduced_holonomy_str_inv not in visited_holonomies:
                visited_holonomies.add(reduced_holonomy_str_inv)

                new_dist_to_tile_sq_inv = _calculate_distance_to_transformed_polygon_sq(
                    p1, new_transform_inv, fundamental_domain_boundary_points, geometry
                )
                heapq.heappush(
                    pq,
                    (
                        new_dist_to_tile_sq_inv,
                        reduced_holonomy_str_inv,
                        new_transform_inv,
                    ),
                )

    # Return the image of p2 in the root tile's coordinate system
    return np.sqrt(min_dist_sq), best_transform @ p2
