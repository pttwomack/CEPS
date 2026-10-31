# mazedemo/test_fixtures.py
"""
Provides test fixtures (complex, reusable test data).
This includes functions to create standard polyhedra and their cut graphs.
"""

import numpy as np
import networkx as nx
import collections
import uuid
from itertools import permutations, combinations
from typing import Optional
from absl import logging

from mazedemo.surface import Surface, FrozenSurface, calculate_chi
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId
from mazedemo.cut import deprecated_api_cut
from mazedemo.grid_fixtures import create_grid
from mazedemo.glue import glue
from mazedemo.generated_fixtures import (
    FLATTENED_GENUS_TWO_FACES,
    FLATTENED_GENUS_TWO_LENGTHS,
)


def create_octahedron(factory: SurfaceFactory) -> SurfaceId:
    """Creates an octahedron in the given factory and returns its SurfaceId."""
    points = np.array(
        [[0, 0, -1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ]
    )

    with factory.update_topology() as surgery:
        workbench = surgery.workbench
        workbench.populate_from_faces(faces.tolist())
        surface_id = surgery.commit("Create octahedron topology")

    with factory.update_properties(surface_id) as annotator:
        with annotator.writer() as view:
            for i, v in enumerate(view.vertices):
                v.pos = points[i]
        annotated_id = annotator.commit("Add coordinates")

    viewer = factory.create_viewer(annotated_id)
    if calculate_chi(viewer) != 2:
        raise RuntimeError("Failed to construct a valid octahedron.")

    return annotated_id


def create_dodecahedron(factory: SurfaceFactory) -> SurfaceId:
    """Creates a dodecahedron in the given factory and returns its SurfaceId."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    points = np.array(
        [
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (-1, -1, 1),
            (1, 1, -1),
            (-1, 1, -1),
            (1, -1, -1),
            (-1, -1, -1),
            (0, phi, 1 / phi),
            (0, phi, -1 / phi),
            (0, -phi, 1 / phi),
            (0, -phi, -1 / phi),
            (1 / phi, 0, phi),
            (-1 / phi, 0, phi),
            (1 / phi, 0, -phi),
            (-1 / phi, 0, -phi),
            (phi, 1 / phi, 0),
            (-phi, 1 / phi, 0),
            (phi, -1 / phi, 0),
            (-phi, -1 / phi, 0),
        ]
    )
    faces = np.array(
        [
            [0, 8, 1, 13, 12],
            [0, 12, 2, 18, 16],
            [0, 16, 4, 9, 8],
            [1, 8, 9, 5, 17],
            [1, 17, 19, 3, 13],
            [2, 10, 11, 6, 18],
            [2, 12, 13, 3, 10],
            [3, 19, 7, 11, 10],
            [4, 14, 15, 5, 9],
            [4, 16, 18, 6, 14],
            [5, 15, 7, 19, 17],
            [6, 11, 7, 15, 14],
        ]
    )

    with factory.update_topology() as surgery:
        workbench = surgery.workbench
        workbench.populate_from_faces(faces.tolist())
        surface_id = surgery.commit("Create dodecahedron topology")

    with factory.update_properties(surface_id) as annotator:
        with annotator.writer() as view:
            for i, v in enumerate(view.vertices):
                v.pos = points[i]
        annotated_id = annotator.commit("Add coordinates")

    viewer = factory.create_viewer(annotated_id)
    if calculate_chi(viewer) != 2:
        raise RuntimeError("Failed to construct a valid dodecahedron.")

    return annotated_id


def create_icosahedron(factory: SurfaceFactory) -> SurfaceId:
    """Creates an icosahedron in the given factory and returns its SurfaceId."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    points = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ]
    )
    points /= np.linalg.norm(points[0])

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )

    with factory.update_topology() as surgery:
        workbench = surgery.workbench
        workbench.populate_from_faces(faces.tolist())
        if not workbench.is_valid():
            raise RuntimeError("icosahedron is not a valid surface")
        surface_id = surgery.commit("Create icosahedron topology")

    with factory.update_properties(surface_id) as annotator:
        with annotator.writer() as view:
            for i, v in enumerate(view.vertices):
                v.pos = points[i]
        annotated_id = annotator.commit("Add coordinates")

    viewer = factory.create_viewer(annotated_id)
    if calculate_chi(viewer) != 2:
        raise RuntimeError("Failed to construct a valid icosahedron.")

    return annotated_id


def create_pair_of_pants(factory: SurfaceFactory) -> SurfaceId:
    """Creates a pair of pants surface and returns its surface id and boundary cycles."""
    surface_id = create_icosahedron(factory)

    cut_graph = nx.Graph()
    cut_graph.add_edges_from(
        ([0, 11], [11, 5], [5, 0], [10, 7], [7, 6], [6, 10], [3, 9], [9, 4], [4, 3])
    )

    piece_graph = deprecated_api_cut(factory, surface_id, cut_graph)

    pants_nodes = [
        n
        for n, d in piece_graph.nodes(data=True)
        if d["type"] == "geometric" and piece_graph.degree(n) == 3
    ]
    pants_id = piece_graph.nodes[pants_nodes[0]]["surface_id"]
    viewer = factory.create_viewer(pants_id)

    if calculate_chi(viewer) != -1:
        raise RuntimeError(
            f"Failed to construct valid pair of pants. Chi was {calculate_chi(viewer)}."
        )
    cycles = [
        [c.cycle_he_id for c in data["corner_cycle"]]
        for _, _, data in piece_graph.edges(pants_nodes, data=True)
    ]

    return pants_id, cycles


def create_genus_two_surface_fixture(factory: SurfaceFactory):
    """Sets up data needed to glue two pairs of pants into a genus-two surface."""
    pants1_id, cycles1 = create_pair_of_pants(factory)
    pants2_id, cycles2 = create_pair_of_pants(factory)

    if len(cycles1) != 3 or len(cycles2) != 3:
        raise RuntimeError(
            "Failed to find three boundary cycles on each pair of pants."
        )
    for i in range(3):
        cycles1[i].sort(key=lambda e: e._local_id)
        cycles2[i].sort(key=lambda e: e._local_id)
    cycles1.sort(key=lambda c: c[0]._local_id)
    cycles2.sort(key=lambda c: c[0]._local_id)

    all_pairings = [
        [
            (cycles1[p[0]][i1], cycles2[0][0]),
            (cycles1[p[1]][i2], cycles2[1][0]),
            (cycles1[p[2]][i3], cycles2[2][0]),
        ]
        for p in permutations([0, 1, 2])
        for i1 in range(3)
        for i2 in range(3)
        for i3 in range(3)
    ]

    return [pants1_id, pants2_id], all_pairings


def create_genus_two_surface(factory: SurfaceFactory) -> SurfaceId:
    ids, all_pairings = create_genus_two_surface_fixture(factory)
    return glue(factory, ids, all_pairings[18])


def create_flattened_genus_two_surface(factory: SurfaceFactory) -> SurfaceId:
    """
    Creates a genus-two surface with a pre-computed flat hyperbolic metric.
    This provides a fast and deterministic way to get a flattened surface for testing.
    The topology and length data are loaded from `generated_fixtures.py`.
    """
    if not FLATTENED_GENUS_TWO_FACES or not FLATTENED_GENUS_TWO_LENGTHS:
        raise RuntimeError(
            "Fixture data is missing. Please run scripts/regenerate_fixtures.py"
        )

    # 1. Create the surface with the exact, pre-computed flattened topology.
    with factory.update_topology() as surgery:
        surgery.workbench.populate_from_faces(FLATTENED_GENUS_TWO_FACES)
        topo_surface_id = surgery.commit("Create pre-flattened genus-two topology")

    # 2. Apply the pre-computed lengths in the canonical order.
    with factory.update_properties(topo_surface_id) as annotation:
        with annotation.writer() as view:
            # Re-create the same canonical sort order used by the generation script.
            sorted_hes = sorted(
                view.half_edges,
                key=lambda he: (he.origin.id._local_id, he.next.origin.id._local_id),
            )

            if len(sorted_hes) != len(FLATTENED_GENUS_TWO_LENGTHS):
                raise ValueError(
                    "Mismatch between number of half-edges in fixture topology and number of lengths."
                )

            for he, length in zip(sorted_hes, FLATTENED_GENUS_TWO_LENGTHS):
                he.length = length

        metric_surface_id = annotation.commit(
            "Apply pre-computed flat hyperbolic metric"
        )

    return metric_surface_id


def create_genus_two_surface_with_edge_lengths(factory: SurfaceFactory) -> SurfaceId:
    """
    Creates a complete, genus-two surface with a flat hyperbolic metric.
    The edge lengths are computed via a robust, iterative conformal map and
    refinement process.
    """

    # lazy load discrete_conformal_map is appropriate,
    # since it isnt part of core surface library.
    # TODO: move this fixture out of surface library.
    from mazedemo.discrete_conformal_map import discrete_conformal_map

    surface_id = create_genus_two_surface(factory)
    viewer = factory.create_viewer(surface_id)

    # Calculate the constant Euclidean edge length of a unit icosahedron.
    t = (1.0 + np.sqrt(5.0)) / 2.0
    p0 = np.array([-1, t, 0])
    p1 = np.array([0, 1, t])
    norm_factor = np.linalg.norm(p0)
    constant_euclidean_length = np.linalg.norm((p0 / norm_factor) - (p1 / norm_factor))

    target_curvature = np.zeros(len(viewer.vertices))

    # Annotate the surface with initial lengths before calling the conformal map function
    with factory.update_properties(surface_id) as annotation:
        with annotation.writer() as view:
            for he in view.half_edges:
                if he.twin:
                    he.length = constant_euclidean_length
        surface_id = annotation.commit("Add initial Euclidean lengths")

    # Compute the flat hyperbolic lengths using the robust method.
    # Updated to not pass initial_lengths explicitly.
    final_id = discrete_conformal_map(
        factory, surface_id, target_curvature=target_curvature
    )

    final_viewer = factory.create_viewer(final_id, custom_properties={"length": None})
    if calculate_chi(final_viewer) != -2:
        raise RuntimeError("Failed to construct a valid genus-two surface.")

    return final_id


def get_octahedron_cut_cases():
    """Returns a dictionary of named cut graphs and their expected properties for the octahedron."""
    return collections.OrderedDict(
        [
            (
                "single_edge",
                (
                    nx.Graph([(1, 2)]),
                    {
                        "num_geom_pieces": 1,
                        "num_topo_pieces": 1,
                        "topo_degrees": [1],
                        "cycle_lens": [2],
                        "geom_chi": [1],
                        "cycle_order": [(1, 2)],
                        "corners": [[3, 3]],
                    },
                ),
            ),
            (
                "path",
                (
                    nx.Graph([(1, 2), (2, 3)]),
                    {
                        "num_geom_pieces": 1,
                        "num_topo_pieces": 1,
                        "topo_degrees": [1],
                        "cycle_lens": [4],
                        "geom_chi": [1],
                        "corners": [[1, 3, 1, 3]],
                    },
                ),
            ),
            (
                "cycle",
                (
                    nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)]),
                    {
                        "num_geom_pieces": 2,
                        "num_topo_pieces": 1,
                        "topo_degrees": [2],
                        "cycle_lens": [4, 4],
                        "geom_chi": [1, 1],
                        "corners": [[3, 3, 3, 3], [1, 1, 1, 1]],
                    },
                ),
            ),
            (
                "two_disjoint_edges",
                (
                    nx.Graph([(1, 2), (3, 4)]),
                    {
                        "num_geom_pieces": 1,
                        "num_topo_pieces": 2,
                        "topo_degrees": [1, 1],
                        "cycle_lens": [2, 2],
                        "geom_chi": [0],
                        "cycle_order": [(1, 2), (3, 4)],
                        "corners": [[3, 3], [3, 3]],
                    },
                ),
            ),
            (
                "dumbbell_graph",
                (
                    nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]),
                    {
                        "num_geom_pieces": 3,
                        "num_topo_pieces": 1,
                        "topo_degrees": [3],
                        "cycle_lens": [3, 3, 8],
                        "geom_chi": [1, 1, 1],
                        "corners": [[2, 2, 1], [1, 1, 1], [2, 2, 2, 2]],
                    },
                ),
            ),
            (
                "star_graph",
                (
                    nx.Graph([(5, 1), (5, 2), (5, 3), (5, 4)]),
                    {
                        "num_geom_pieces": 1,
                        "num_topo_pieces": 1,
                        "topo_degrees": [1],
                        "cycle_lens": [8],
                        "geom_chi": [1],
                        "cycle_order": [(1, 5, 2, 5, 3, 5, 4, 5)],
                        "corners": [[3, 0, 3, 0, 3, 0, 3, 0]],
                    },
                ),
            ),
        ]
    )


def get_dodecahedron_cut_cases():
    """Returns a dictionary of named cut graphs for the dodecahedron."""
    perfect_matching_edges = [
        (0, 8),
        (1, 17),
        (2, 10),
        (3, 19),
        (4, 14),
        (5, 9),
        (6, 11),
        (7, 15),
        (12, 13),
        (16, 18),
    ]
    cases = {"10_punctures": nx.Graph(perfect_matching_edges)}
    return cases


def create_donut_surface(
    factory: SurfaceFactory,
    R: float = 2.0,
    r: float = 1.0,
    rows: int = 3,
    cols: int = 3,
    slice_around: bool = False,
    slice_through: bool = False,
) -> SurfaceId:
    """
    Creates a grid surface and gives it a non-flat metric by embedding it on a 3D torus.
    By default, creates a torus. Flags allow slicing it into an annulus or a rectangle.
    """
    # Create a grid topology using the general public grid fixture.
    # This surface now has `grid_row` and `grid_col` properties on its vertices.
    surface_id = create_grid(
        factory,
        rows=rows,
        cols=cols,
        wrap_x=not slice_around,
        wrap_y=not slice_through,
    )

    # Assign a non-flat metric by calculating edge lengths from a 3D torus embedding.
    # This overwrites the initial flat metric lengths.
    with factory.update_properties(surface_id) as annotation:
        with annotation.writer() as view:
            positions_3d = {}
            # The custom_properties argument makes the new properties available via attributes.
            custom_view = factory.create_viewer(
                surface_id, custom_properties={"grid_row": None, "grid_col": None}
            )
            for v in custom_view.vertices:
                # This logic is now robust, reading properties instead of using _local_id.
                i = v.grid_row
                j = v.grid_col
                theta = 2 * np.pi * i / rows
                phi = 2 * np.pi * j / cols
                x = (R + r * np.cos(theta)) * np.cos(phi)
                y = (R + r * np.cos(theta)) * np.sin(phi)
                z = r * np.sin(theta)
                positions_3d[v.id] = np.array([x, y, z])

            # Use the original `view` for writing, as it's the correct context.
            for he in view.half_edges:
                p1 = positions_3d[he.origin.id]
                p2 = positions_3d[he.next.origin.id]
                he.length = np.linalg.norm(p1 - p2)
                if he.twin:
                    he.twin.length = he.length

        length_surface_id = annotation.commit(
            "Add initial Euclidean edge lengths from embedding"
        )

    return length_surface_id


def create_non_flat_annulus(
    factory: SurfaceFactory, rows: int = 3, cols: int = 4
) -> SurfaceId:
    """
    Creates a triangulated grid forming an annulus (a cylinder) and gives it a
    non-flat metric derived from embedding it as a patch on a 3D torus.
    """
    return create_donut_surface(factory, rows=rows, cols=cols, slice_through=True)


def create_torus_fundamental_domain(factory: SurfaceFactory):
    """
    Creates a curved torus, flattens it, and computes its fundamental domain.

    Returns:
        A tuple containing the flattened surface ID and a tuple of domain artifacts.
    """
    # Lazy imports to prevent circular dependencies
    from mazedemo.discrete_conformal_map import discrete_conformal_map
    from mazedemo.fundamental_domain import compute_fundamental_domain
    from mazedemo.euclidean import EuclideanGeometry

    curved_torus_id = create_donut_surface(factory)
    view = factory.create_viewer(curved_torus_id)
    target_curvature = np.zeros(len(view.vertices))

    # The optimizer returns only the surface ID when return_gradient=False
    flat_torus_id = discrete_conformal_map(factory, curved_torus_id, target_curvature)

    geometry = EuclideanGeometry()

    domain_artifacts = compute_fundamental_domain(factory, flat_torus_id, geometry)

    return flat_torus_id, domain_artifacts
