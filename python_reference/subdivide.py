from typing import Any
from absl import logging

from mazedemo.ids import HalfEdgeId, SurfaceId, VertexId
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.surface import Surface, Face, HalfEdge
from mazedemo.surface_util import validate_topology, get_topology_str


def bisect_edge(he: HalfEdge, workbench: Surface):
    """Inserts a vertex betwee the endpoints of the half_edge (and its twin)."""
    mid = workbench.new_vertex()
    new_he = workbench.new_half_edge(origin=mid, parent_ids=he.parent_ids)
    new_he.face = he.face
    new_he.next = he.next
    new_he.prev = he
    he.next.prev = new_he
    he.next = new_he
    mid.half_edge = new_he
    if he.twin:
        new_he.twin = workbench.new_half_edge(
            origin=he.twin.origin, parent_ids=he.twin.parent_ids
        )
        new_he.twin.twin = new_he
        new_he.twin.face = he.twin.face
        new_he.twin.next = he.twin
        new_he.twin.prev = he.twin.prev
        he.twin.prev.next = new_he.twin
        he.twin.prev = new_he.twin
        he.twin.origin.half_edge = new_he.twin
        he.twin.origin = mid


def subdivide_face(face: Face, workbench: Surface):
    """Subdivides hexagonal face into 4 triangles. Original face in center."""
    corner_hes = [he for he in face.half_edges if he.origin.parent_ids]
    if len(face.half_edges) != 6:
        raise ValueError(
            f"original face started with {len(face.half_edges)} half_edges"
        )
    for corner_he in corner_hes:
        prev_he = corner_he.prev
        # create new elements
        new_face = workbench.new_face(parent_ids=face.parent_ids)
        new_he = workbench.new_half_edge(origin=corner_he.next.origin)
        new_he.twin = workbench.new_half_edge(origin=prev_he.origin)
        new_he.twin.twin = new_he

        # (re)link faces
        new_he.twin.face = face
        face.half_edge = new_he.twin
        new_he.face = corner_he.face = prev_he.face = new_face
        new_face.half_edge = corner_he

        # link new twin at both ends.
        new_he.twin.next = corner_he.next
        prev_he.prev.next = new_he.twin
        new_he.twin.prev = prev_he.prev
        corner_he.next.prev = new_he.twin

        # link new he at both ends.
        new_he.next = prev_he
        new_he.prev = corner_he
        prev_he.prev = new_he
        corner_he.next = new_he
        if len(new_face.half_edges) != 3:
            logging.error(
                "created neww face with %d half_edges", len(new_face.half_edges)
            )
        if len(face.half_edges) > 5:
            logging.error("original face still has %d half_edges", len(face.half_edges))


def subdivide(
    factory: SurfaceFactory, surface_id: SurfaceId, geometry: Any
) -> SurfaceId:
    """
    Subdivides each triangle of a surface into four by performing a 1-to-4 split.

    This function manually constructs the new, refined topology and then annotates
    it with new edge lengths. It is designed to be robust against parallel edges
    and self-loops by using correct provenance and local maps for construction.
    """
    # --- 1. Create new topology in a single surgery ---
    with factory.update_topology(source_ids=[surface_id]) as surgery:
        workbench = surgery.workbench

        # Bisect all edges, creating hexagonal faces.
        old_hes = [he for he in workbench.half_edges]  # avoid re-update
        for he in old_hes:
            if he.next.origin.parent_ids:
                if not he.twin or he.twin.id > he.id:
                    bisect_edge(he, workbench)

        # subdivide all faces
        old_faces = [face for face in workbench.faces]
        for face in old_faces:
            subdivide_face(face, workbench)

        for face in workbench.faces:
            if len(face.half_edges) != 3:
                raise ValueError("not all faces are triangles")

        if not validate_topology(workbench):
            logging.error("Topology invalid after face subdivision")
            logging.error("Surface state:\n%s", get_topology_str(workbench))
            raise ValueError("Topology invalid after face subdivision")

        new_id = surgery.commit("Subdivide surface")

    # --- 2. Annotate new surface with edge lengths ---
    with factory.update_properties(new_id) as annotation:
        new_viewer = factory.create_viewer(new_id)

        with annotation.writer() as view:
            # Divide all assigned edge lengths by 2.
            for he in view.half_edges:
                if he.length:
                    he.length = he.length / 2

            # compute lengths of new edges.
            for face in view.faces:
                # original faces have no assigned lengths.
                if all(not he.length for he in face.half_edges):
                    new_lengths = [he.twin.next.length for he in face.half_edges]
                    orig_lengths = [2 * l for l in new_lengths]
                    angles = geometry.compute_angles_from_opposite_lengths(orig_lengths)
                    for i, he in enumerate(face.half_edges):
                        he.length = he.twin.length = (
                            geometry.compute_length_from_corner(
                                angles[i],
                                new_lengths[(i + 1) % 3],
                                new_lengths[(i + 2) % 3],
                            )
                        )
                else:
                    if not all(he.length for he in face.half_edges):
                        logging.error("face %s has missing lengths", face.id)
                        for he in face.half_edges:
                            logging.info("he %s as length=%s", he.id, he.length)

        new_id_with_lengths = annotation.commit("Add subdivided lengths")

    return new_id_with_lengths
