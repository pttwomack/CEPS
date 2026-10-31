# mazedemo/mazeutil.py
"""
This module provides utilities for maze creation.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque
from scipy.spatial import KDTree
import triangle
from absl import logging

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId
from mazedemo.viewer import Viewer
from mazedemo.hyperbolic import (
    construct_triangle,
    get_isometry_aligning_edges,
    distance as hyperbolic_distance,
    lorentz_dot,
)
from mazedemo.cut import cut
from mazedemo.glue import glue


@dataclass
class Wrap:
    """A wrapper to allow np.random.choice on a list of arbitrary objects."""

    o: object


def ust(g: nx.Graph) -> nx.Graph:
    """
    Computes a uniform spanning tree of a graph using Wilson's algorithm.
    """
    if not list(g.nodes):
        return nx.Graph()
    # np.random.choice requires a list of integers or a wrapped list of objects.
    target = {np.random.choice([Wrap(n) for n in g.nodes]).o}
    st = nx.Graph()
    while True:
        remaining = frozenset(g.nodes) - target
        if not remaining:
            break
        current = np.random.choice([Wrap(n) for n in remaining]).o
        visited, path = set(), []
        while current not in target:
            visited.add(current)
            if not list(g.edges(current)):
                break
            e = np.random.choice([Wrap(e) for e in g.edges(current, data=True)]).o
            neighbor = e[1] if e[0] == current else e[0]
            path.append((current, neighbor, e[2]))
            current = neighbor
            if current in visited:
                while path and path[-1][1] != current:
                    visited.remove(path.pop()[1])
                if path:
                    visited.remove(path.pop()[1])
        st.add_edges_from(path)
        target |= visited
    return st


def annotate_lengths_from_positions(
    factory: SurfaceFactory, surface_id: SurfaceId
) -> SurfaceId:
    """
    Annotates each half-edge of a surface with a 'length' property, calculated
    as the Euclidean distance between the 'position' properties of its endpoints.

    Args:
        factory: The SurfaceFactory to use.
        surface_id: The ID of the surface to annotate. It is assumed that the
                    vertices of this surface have a 'position' property.

    Returns:
        The ID of the new surface with 'length' annotations.
    """
    viewer = factory.create_viewer(surface_id, custom_properties={"position": None})

    with factory.update_properties(surface_id) as annotation:
        with annotation.writer() as writer_view:
            for he in viewer.half_edges:
                # Process each edge only once to avoid redundant calculations.
                if he.twin and he.id > he.twin.id:
                    continue

                origin_pos = he.origin.position
                dest_pos = he.next.origin.position

                length = 1.0  # Default length for edges connected to infinity.
                # Calculate Euclidean length only if both endpoints are 2D points.
                if (
                    isinstance(origin_pos, tuple)
                    and len(origin_pos) == 2
                    and isinstance(dest_pos, tuple)
                    and len(dest_pos) == 2
                ):
                    length = np.linalg.norm(np.array(origin_pos) - np.array(dest_pos))

                # Set the length on the half-edge and its twin.
                he_proxy = writer_view.get_proxy(he.id)
                if he_proxy:
                    he_proxy.length = length

                if he.twin:
                    twin_proxy = writer_view.get_proxy(he.twin.id)
                    if twin_proxy:
                        twin_proxy.length = length

        annotated_surface_id = annotation.commit(
            "Annotate edges with Euclidean lengths"
        )

    return annotated_surface_id


def triangulate_complement(factory: SurfaceFactory, g: nx.Graph) -> nx.Graph:
    """
    Creates a triangulation of the sphere containing the graph g, then cuts
    it along the edges of g. This is done by creating a triangulated disk from
    the input graph, creating a corresponding "cap" of triangles, and then
    gluing the two pieces together to form a sphere.

    Args:
        factory: The SurfaceFactory to use for creating surfaces.
        g: A networkx graph where nodes are 2D positions.

    Returns:
        A networkx graph representing the partitioned surface after the cut.
    """
    if not g.nodes:
        return nx.Graph()

    positions = list(g.nodes)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    if len(positions) < 3:
        return nx.Graph()

    # Define the planar straight-line graph (PSLG) for the constrained triangulation.
    pslg = {
        "vertices": np.array(positions),
        "segments": np.array([[pos_to_idx[u], pos_to_idx[v]] for u, v in g.edges]),
    }

    # Perform the constrained Delaunay triangulation.
    t = triangle.triangulate(pslg, "pc")
    simplices = t["triangles"]
    all_disk_vertices = t["vertices"]

    # --- Step 1: Create the planar disk from the CDT output ---
    with factory.update_topology() as surgery:
        surgery.workbench.populate_from_faces(simplices.tolist())
        disk_id = surgery.commit("Create triangulated disk")

    # --- Step 1a: Annotate the disk with vertex positions ---
    with factory.update_properties(disk_id) as annotation:
        with annotation.writer() as view:
            # The vertices created by populate_from_faces correspond to the rows
            # in the `all_disk_vertices` array from the triangulation output.
            for i, pos in enumerate(all_disk_vertices):
                v_proxy = view.get_proxy(VertexId(disk_id, i))
                v_proxy.position = tuple(pos)
                logging.vlog(2, "set position of %s to %s", v_proxy.id, pos)
        disk_id = annotation.commit("Add vertex positions to disk")

    disk_view = factory.create_viewer(disk_id)
    disk_boundary_hes = [he for he in disk_view.half_edges if he.twin is None]

    if not disk_boundary_hes:
        # The "disk" was already a sphere (e.g., from 3 input points).
        sphere_id = disk_id
    else:
        # --- Step 2: Create a "cap" surface to glue onto the disk ---
        num_boundary_verts = len(disk_boundary_hes)

        # Create faces for a fan of triangles around a central point (index 0).
        cap_faces = []
        for i in range(num_boundary_verts):
            v_inf_idx = 0
            v_i_idx = i + 1
            v_i_plus_1_idx = ((i + 1) % num_boundary_verts) + 1
            # Note the order to ensure consistent orientation with the disk.
            cap_faces.append([v_inf_idx, v_i_plus_1_idx, v_i_idx])

        with factory.update_topology() as cap_surgery:
            cap_surgery.workbench.populate_from_faces(cap_faces)
            cap_id = cap_surgery.commit("Create infinity cap")

        # --- Step 3: Glue the disk and the cap together ---
        # Find a starting half-edge on each boundary to seed the glue operation.
        disk_start_he = next(he for he in disk_view.half_edges if he.twin is None)

        cap_view = factory.create_viewer(cap_id)
        cap_start_he = next(he for he in cap_view.half_edges if he.twin is None)

        # The glue function only needs one pair of corresponding half-edges.
        glue_pairing = [(disk_start_he.id, cap_start_he.id)]

        sphere_id = glue(factory, [disk_id, cap_id], glue_pairing)

    # --- Step 4: Annotate the point-at-infinity vertex ---
    # The 'position' property from the disk vertices has propagated through the
    # glue operation. We now robustly find the vertex that descended from the
    # cap's central point and explicitly set its position to None.
    if not disk_boundary_hes:
        # The original disk was already a closed sphere, no cap was added.
        pass
    else:
        # Find the descendant of the cap's central vertex (index 0) using its topological history.
        cap_v0_id = VertexId(cap_id, 0)
        trace_results = factory.trace_elements([sphere_id], {cap_v0_id})
        descendants = trace_results.get(cap_v0_id)

        if descendants:
            v_at_infinity_id = next(iter(descendants))
            with factory.update_properties(sphere_id) as annotation:
                with annotation.writer() as view:
                    v_proxy = view.get_proxy(v_at_infinity_id)
                    if v_proxy:
                        v_proxy.position = (0.0, 0.0, 1.0)
                sphere_id = annotation.commit(
                    "Set position for point-at-infinity vertex"
                )

    # --- Final Step: Identify cut edges and perform the cut ---
    sphere_view = factory.create_viewer(sphere_id, custom_properties={"position": None})

    # Build a map from position to VertexId by reading the propagated 'position' property.
    final_pos_to_vid = {}
    for v in sphere_view.vertices:
        try:
            # The custom property getter for 'position' raises an AttributeError
            # if the property value is None (e.g., for the point at infinity).
            pos = v.position
            if pos is not None:
                final_pos_to_vid[pos] = v.id
        except AttributeError:
            # Skip vertices where 'position' is not set or is None.
            pass

    # Build a lookup map for efficient edge finding on the sphere.
    edge_map = {}
    for he in sphere_view.half_edges:
        # To avoid duplicates, only process one of the two half-edges for an edge.
        if he.twin is None or he.id < he.twin.id:
            edge_key = frozenset({he.origin.id, he.next.origin.id})
            edge_map[edge_key] = he.id

    # Use the maps to find the half-edges on the sphere that correspond to the input graph edges.
    cut_hes = set()
    for u_pos, v_pos in g.edges:
        u_vid = final_pos_to_vid.get(u_pos)
        v_vid = final_pos_to_vid.get(v_pos)

        if u_vid is None or v_vid is None:
            raise ValueError(
                f"Graph edge ({u_pos}, {v_pos}) contains nodes not found on the sphere."
            )

        edge_key = frozenset({u_vid, v_vid})
        he_id = edge_map.get(edge_key)

        if he_id is None:
            raise RuntimeError(
                f"Graph edge between {u_pos} and {v_pos} was not found in the final triangulation."
            )
        cut_hes.add(he_id)

    return cut(factory, sphere_id, cut_hes)
