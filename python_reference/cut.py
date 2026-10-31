# mazedemo/cut.py
"""
Implements a `cut` operation that can be used to perform topological surgery
on a manifold mesh. This version uses a declarative, transactional approach
where each step of the algorithm is an atomic commit.
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple, Set, Any, Hashable
import copy
from absl import logging
import collections, dataclasses

from mazedemo.surface import Surface, calculate_chi
from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId
from mazedemo.surgery import Surgery
from mazedemo.viewer import Viewer


# Forward declaration to avoid circular import
class SurfaceFactory:
    pass


@dataclasses.dataclass(frozen=True)
class _Corner:
    """
    A dataclass to hold information about a corner of a geometric piece.
    Stores the apex vertex and the half-edge ids of all edges originating
    at the apex that belong in the corner.
    """

    apex_v_id: VertexId
    cycle_he_id: HalfEdgeId
    fan_he_ids: Tuple[HalfEdgeId, ...]


# --- Internal Read-Only Analysis Helpers (Using Viewer API) ---


def _find_topology(
    view: Viewer, g_hes: Set[HalfEdgeId], cut_graph: nx.Graph
) -> nx.Graph:
    """
    Phase 1: Determines the topological pieces resulting from a cut.
    """
    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(view.faces)
    for he in view.half_edges:
        # If a half-edge is not in the cut set, it connects two faces in the dual graph.
        if he.face and he.twin and he.twin.face and he.id not in g_hes:
            dual_graph.add_edge(he.face, he.twin.face)

    piece_graph, face_to_geom_id, piece_id_counter = nx.Graph(), {}, 0
    for i, comp_faces in enumerate(nx.connected_components(dual_graph)):
        geom_id = piece_id_counter
        piece_id_counter += 1
        piece_graph.add_node(
            geom_id, type="geometric", faces=[f.id for f in comp_faces]
        )
        for face in comp_faces:
            face_to_geom_id[face.id] = geom_id

    # The cut_graph provides connectivity for the cut path, used to group g_hes into components.
    for i, comp_vids in enumerate(nx.connected_components(cut_graph)):
        topo_id = piece_id_counter
        piece_id_counter += 1

        comp_hes = {
            he_id for he_id in g_hes if view.get_proxy(he_id).origin.id in comp_vids
        }
        piece_graph.add_node(topo_id, type="topological", half_edges=comp_hes)

        adjacent_geom_ids = {
            face_to_geom_id[view.get_proxy(he_id).face.id]
            for he_id in comp_hes
            if view.get_proxy(he_id).face
        }
        for geom_id in adjacent_geom_ids:
            piece_graph.add_edge(geom_id, topo_id)

    return piece_graph


def _analyze_interface(
    view: Viewer, geom_face_ids: List[FaceId], topo_he_ids: Set[HalfEdgeId]
) -> List[_Corner]:
    """Analyzes the interface using the read-only Viewer API."""
    interface_hes = {
        he
        for face_id in geom_face_ids
        for he in view.get_proxy(face_id).half_edges
        if he.twin and he.twin.id in topo_he_ids
    }
    if not interface_hes:
        return []

    start_he = next(iter(interface_hes))
    he_cycle, curr = [start_he], start_he
    while len(he_cycle) < len(interface_hes):
        next_he = curr.next
        while not (next_he.twin and next_he.twin.id in topo_he_ids):
            twin = next_he.twin
            if not twin:
                raise RuntimeError(f"Interface traversal failed at edge {next_he.id}")
            next_he = twin.next
        he_cycle.append(next_he)
        curr = next_he
    if len(he_cycle) != len(interface_hes):
        raise RuntimeError("Interface analysis failed: not a single, simple cycle.")

    corner_cycle = []
    for i, he_out in enumerate(he_cycle):
        he_in = he_cycle[i - 1]

        apex_v_id = he_out.origin.id

        fan_hes = []
        curr_fan = he_in.next
        while curr_fan != he_out:
            twin = curr_fan.twin
            if not twin:
                raise RuntimeError(
                    f"Fan walk failed: interior edge {curr_fan} has no twin."
                )
            fan_hes.append(curr_fan)
            curr_fan = twin.next

        assert all(he_out.origin == fan_he.origin for fan_he in fan_hes)
        corner_cycle.append(
            _Corner(
                apex_v_id=apex_v_id,
                cycle_he_id=he_out.id,
                fan_he_ids=tuple([he.id for he in fan_hes]),
            )
        )
    return corner_cycle


# --- Transactional Mutation Helpers ---


def _partition_piece(
    factory: SurfaceFactory, source_id: SurfaceId, faces_to_keep: List[FaceId]
) -> SurfaceId:
    """Creates a new surface containing only the specified faces."""
    view = factory.create_viewer(source_id)
    footprint_hes = {
        he.id for face_id in faces_to_keep for he in view.get_proxy(face_id).half_edges
    }
    footprint_vs = {he.origin.id for he in view.half_edges if he.id in footprint_hes}

    footprint = {
        "face": set(faces_to_keep),
        "half_edge": footprint_hes,
        "vertex": footprint_vs,
    }

    with factory.update_topology([source_id], footprint=footprint) as surgery:
        surgery.set_as_partition()
        return surgery.commit(f"Partitioned piece from {source_id}")


def _unravel_one_cycle(
    factory: SurfaceFactory, surface_id: SurfaceId, corner_cycle: List[_Corner]
) -> SurfaceId:
    """
    Performs the unravelling operation for a single boundary cycle.
    This clones the vertices along the cut, untwins the half-edges, and
    re-parents the relevant edges to form a new, unsealed boundary.
    """
    # 1. Build the footprint for the surgery.
    ancestor_he_ids = {c.cycle_he_id for c in corner_cycle}
    ancestor_he_ids.update({fid for c in corner_cycle for fid in c.fan_he_ids})
    ancestor_v_ids = {c.apex_v_id for c in corner_cycle}

    footprint = {"half_edge": ancestor_he_ids, "vertex": ancestor_v_ids}

    # 2. Start the surgery.
    with factory.update_topology(
        [surface_id], trace_ids=ancestor_he_ids | ancestor_v_ids
    ) as surgery:
        workbench = surgery.workbench

        verts_to_delete = set()

        # 3. Process each corner to split vertices and re-parent edges.
        for corner in corner_cycle:
            # Identify the edge and the original vertex *on the workbench*
            cycle_he = workbench.get_descendant(corner.cycle_he_id)
            original_v = cycle_he.origin
            verts_to_delete.add(original_v)

            # Create a NEW vertex for this specific corner (splitting the original)
            new_v = workbench.new_vertex(parent_ids=[corner.apex_v_id])

            # Re-parent the fan of interior edges to this new vertex
            for fan_he_id in corner.fan_he_ids:
                workbench.get_descendant(fan_he_id).origin = new_v

            # Re-parent the outgoing cycle edge
            cycle_he.origin = new_v
            new_v.half_edge = cycle_he

            # 4. Un-twin the half-edges along the boundary.
            if cycle_he.twin:
                cycle_he.twin.twin = None
            cycle_he.twin = None

        # 5. Delete the original vertices.
        # Note: If the cut logic is correct, the original vertex has been completely
        # stripped of its edges and moved to the new split vertices.
        for v in verts_to_delete:
            workbench.delete_vertex(v)

        return surgery.commit(f"Unravelled boundary on {surface_id}")


# --- Public API Functions ---


def cut(
    factory: SurfaceFactory, source_id: SurfaceId, cut_edges: Set[HalfEdgeId]
) -> nx.Graph:
    """
    Cuts a surface along a specified set of half-edges.

    Args:
        factory: The SurfaceFactory instance.
        source_id: The ID of the surface to cut.
        cut_edges: A set of HalfEdgeIds to cut along. Does not matter whether or not
               edges are specified by single HalfEdgeId, or twins are included.
    """
    view = factory.create_viewer(source_id)

    # Ensure cut_edges is symmetric: if he is in g_hes, he.twin must also be.
    symmetric_cut_edges = set(cut_edges)
    for he_id in cut_edges:
        he = view.get_proxy(he_id)
        if he and he.twin:
            symmetric_cut_edges.add(he.twin.id)

    if not symmetric_cut_edges:
        logging.warning("Cut graph is empty or contains no edges found on the surface.")
        piece_graph = nx.Graph()
        piece_graph.add_node(
            0, type="geometric", surface_id=source_id, chi=calculate_chi(view)
        )
        return piece_graph

    # Build the vertex graph of the cut path for finding connected components.
    cut_graph = nx.Graph()
    for he_id in symmetric_cut_edges:
        he = view.get_proxy(he_id)
        if he.twin:
            cut_graph.add_edge(he.origin.id, he.twin.origin.id)

    # 1. Read-only analysis phase using the Viewer.
    piece_graph = _find_topology(view, symmetric_cut_edges, cut_graph)

    # 2. Analyze all interfaces using the Viewer.
    for u, v, data in piece_graph.edges(data=True):
        if piece_graph.nodes[u]["type"] == piece_graph.nodes[v]["type"]:
            raise RuntimeError("piece graph is not bipartite")
        if piece_graph.nodes[u]["type"] == "geometric":
            geom_node_id, topo_node_id = u, v
        else:
            geom_node_id, topo_node_id = v, u

        data["corner_cycle"] = _analyze_interface(
            view,
            piece_graph.nodes[geom_node_id]["faces"],
            piece_graph.nodes[topo_node_id]["half_edges"],
        )

    # 3. Transactional mutation phase for each piece.
    for node_id, data in piece_graph.nodes(data=True):
        if data["type"] == "geometric":
            faces_to_keep_ids = data["faces"]

            current_piece_id = _partition_piece(factory, source_id, faces_to_keep_ids)

            corner_cycles = [
                d["corner_cycle"]
                for _, _, d in piece_graph.edges(node_id, data=True)
                if "corner_cycle" in d
            ]
            for corner_cycle in corner_cycles:
                if corner_cycle:
                    current_piece_id = _unravel_one_cycle(
                        factory, current_piece_id, corner_cycle
                    )

            data["surface_id"] = current_piece_id
            data["chi"] = calculate_chi(factory.create_viewer(current_piece_id))

    return piece_graph


def deprecated_api_cut(
    factory: SurfaceFactory, source_id: SurfaceId, g: nx.Graph
) -> nx.Graph:
    """
    DEPRECATED: Performs a cut operation by orchestrating a series of atomic transactions.
    This wrapper translates the old, brittle API to the new internal logic.
    """
    view = factory.create_viewer(source_id)

    # --- Translation from old API to new internal API ---
    local_id_to_vid = {v.id._local_id: v.id for v in view.vertices}

    he_map = collections.defaultdict(list)
    for he in view.half_edges:
        if he.twin and he.id._local_id < he.twin.id._local_id:
            u_id = he.origin.id
            v_id = he.twin.origin.id
            he_map[frozenset({u_id, v_id})].append(he.id)

    g_hes = set()
    for u_local, v_local in g.edges():
        u_vid = local_id_to_vid.get(u_local)
        v_vid = local_id_to_vid.get(v_local)
        if u_vid is None or v_vid is None:
            raise ValueError(
                f"Graph edge ({u_local}, {v_local}) contains vertices not found on the surface."
            )

        edge_key = frozenset({u_vid, v_vid})
        matching_he_ids = he_map.get(edge_key, [])

        if len(matching_he_ids) > 1:
            raise ValueError(
                f"Ambiguous cut: Found {len(matching_he_ids)} parallel edges between "
                f"vertices {u_vid} and {v_vid}. Please use the new `cut` API and "
                "specify the exact HalfEdgeId to cut."
            )
        elif matching_he_ids:
            he_id = matching_he_ids[0]
            g_hes.add(he_id)
        else:
            raise ValueError(
                f"Could not find half-edge for graph edge {u_local}-{v_local}"
            )

    return cut(factory, source_id, g_hes)
