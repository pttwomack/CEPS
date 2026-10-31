# mazedemo/fundamental_domain.py
"""
Implements the construction of a fundamental domain for a closed, orientable
surface, using a provided geometry object.
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Any
from absl import logging
import collections

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId, FaceId, HalfEdgeId, VertexId
from mazedemo.viewer import Viewer, Face, Vertex


def _get_holonomy_inverse(h_class: Any) -> Any:
    """Computes the inverse of a symbolic holonomy class."""
    if isinstance(h_class, str) and h_class:
        return "".join(c.swapcase() for c in reversed(h_class))
    return h_class  # Trivial holonomy ('') is its own inverse.


def _reduce_class(class_str: str) -> str:
    """Reduces a symbolic holonomy class string by cancelling generators."""
    reduced = class_str
    for _ in range(len(class_str) // 2 + 1):
        prev = reduced
        for gen in "abcdefghijklmnopqrstuvwxyz":
            reduced = reduced.replace(gen + gen.upper(), "")
            reduced = reduced.replace(gen.upper() + gen, "")
        if reduced == prev:
            break
    return reduced


def _classify_holonomy(
    view: Viewer, spanning_tree: nx.Graph, dual_graph: nx.Graph
) -> Tuple[List[str], Dict[HalfEdgeId, Any]]:
    """
    Classifies the holonomy of all seams via vertex-based propagation.
    """
    holonomy_classes: Dict[HalfEdgeId, Any] = {he.id: None for he in view.half_edges}

    # Initialize seams in the spanning tree as trivial holonomy.
    for u, v in spanning_tree.edges():
        he_id = dual_graph[u][v]["half_edge"]
        he = view.get_proxy(he_id)
        holonomy_classes[he.id] = ""
        holonomy_classes[he.twin.id] = ""

    unassigned_seams = {
        he.id for he in view.half_edges if holonomy_classes[he.id] is None
    }
    holonomy_generator = (chr(ord("a") + i) for i in range(26))
    generators = []

    # The queue contains vertices that might be solvable. Initially, it's all vertices.
    queue = collections.deque(view.vertices)

    # We are done when the queue is empty AND we have no more unassigned seams to seed.
    while queue or unassigned_seams:
        # If the queue is empty but seams remain, the graph is disconnected from
        # the classified region. Introduce a new generator to cut a cycle and
        # restart propagation.
        if not queue:
            try:
                # Find an unassigned seam that is an interior edge.
                seed_id = next(
                    he_id
                    for he_id in unassigned_seams
                    if view.get_proxy(he_id).twin is not None
                )
            except StopIteration:
                # All remaining unassigned seams are boundary edges. We are done.
                break
            seed_he = view.get_proxy(seed_id)
            new_symbol = next(holonomy_generator)
            generators.append(new_symbol)

            holonomy_classes[seed_id] = new_symbol
            holonomy_classes[seed_he.twin.id] = _get_holonomy_inverse(new_symbol)
            unassigned_seams.discard(seed_id)
            unassigned_seams.discard(seed_he.twin.id)

            # Add endpoints to the queue to restart propagation.
            if seed_he.origin:
                queue.append(seed_he.origin)
            if seed_he.twin and seed_he.twin.origin:
                queue.append(seed_he.twin.origin)
            continue

        vertex = queue.popleft()
        start_he = vertex.half_edge
        if not start_he:
            continue

        # Collect all outgoing half-edges in a cycle around the vertex.
        outgoing_hes = [start_he]
        curr = start_he
        while curr.prev.twin and curr.prev.twin != start_he:
            curr = curr.prev.twin
            outgoing_hes.append(curr)

        # Ensure we have a full, closed fan of edges before proceeding.
        if not curr.prev.twin or curr.prev.twin != start_he:
            continue

        unclassified = [h for h in outgoing_hes if holonomy_classes[h.id] is None]

        # If exactly one edge is unclassified, we can solve for its holonomy.
        if len(unclassified) == 1:
            h_unknown = unclassified[0]
            # The product of holonomies around a vertex is trivial.
            # C_0 * C_1 * ... * C_{n-1} = id
            # We need to solve for C_k.
            # C_k = (C_{k+1} * ... * C_{n-1} * C_0 * ... * C_{k-1})^{-1}
            try:
                k = outgoing_hes.index(h_unknown)
            except ValueError:
                continue

            # Build the product string C_{k+1}*...*C_{n-1}*C_0*...*C_{k-1}
            product_list = outgoing_hes[k + 1 :] + outgoing_hes[:k]
            product_str = "".join(holonomy_classes[h.id] for h in product_list)

            # The new class is the inverse of this product.
            new_class = _get_holonomy_inverse(product_str)
            new_class = _reduce_class(new_class)

            holonomy_classes[h_unknown.id] = new_class
            holonomy_classes[h_unknown.twin.id] = _get_holonomy_inverse(new_class)
            unassigned_seams.discard(h_unknown.id)
            unassigned_seams.discard(h_unknown.twin.id)

            # The neighbor vertex might now be solvable. Add it to the queue.
            if h_unknown.twin and h_unknown.twin.origin:
                queue.append(h_unknown.twin.origin)
            # Re-queue the current vertex, as it might be able to solve for another edge.
            queue.append(vertex)

    return generators, holonomy_classes


def _layout_domain(
    view: Viewer,
    spanning_tree: nx.Graph,
    dual_graph: nx.Graph,
    centered_embeddings: Dict[FaceId, Dict[VertexId, np.ndarray]],
    isometries: Dict[HalfEdgeId, np.ndarray],
    geometry: Any,
):
    """
    Places the faces of the fundamental domain in a global coordinate system.
    """
    start_face_id = view.faces[0].id
    global_transforms = {start_face_id: geometry.identity_transform()}
    placed_triangles = {start_face_id: centered_embeddings[start_face_id]}

    queue = collections.deque([start_face_id])
    visited = {start_face_id}

    while queue:
        u_id = queue.popleft()

        for v_id in spanning_tree.neighbors(u_id):
            if v_id not in visited:
                visited.add(v_id)

                edge_he_id = dual_graph[u_id][v_id]["half_edge"]
                edge_he = view.get_proxy(edge_he_id)

                # We need the transform from the child's (v_id) coordinate system
                # to the parent's (u_id) coordinate system. Let's call it T_v_to_u.
                # The global transform for v is then T_u_to_global @ T_v_to_u.
                if edge_he.face.id == u_id:
                    # he is in u, he.twin is in v.
                    # isometries[he.id] is T_v_to_u.
                    local_isometry = isometries[edge_he.id]
                else:  # edge_he.face.id == v_id
                    # he is in v, he.twin is in u.
                    # isometries[he.twin.id] is T_v_to_u.
                    local_isometry = isometries[edge_he.twin.id]

                # The new global transform is parent_global @ local_v_to_u
                global_transforms[v_id] = global_transforms[u_id] @ local_isometry
                assert geometry.is_isometry(global_transforms[v_id])

                placed_triangles[v_id] = {
                    vid: global_transforms[v_id] @ pos
                    for vid, pos in centered_embeddings[v_id].items()
                }

                queue.append(v_id)

    return placed_triangles, global_transforms


def _build_weighted_dual_graph_and_spanning_tree(
    view: Viewer, centered_embeddings, isometries, geometry
):
    """Builds the weighted dual graph and computes the shortest-path spanning tree."""
    dual_graph = nx.Graph()
    for he in view.half_edges:
        if he.twin and he.face and he.twin.face:
            f1_id, f2_id = he.face.id, he.twin.face.id

            dist = geometry.calculate_dual_edge_weight(
                centered_embeddings[f1_id],
                centered_embeddings[f2_id],
                isometries[he.id],  # Isometry from f2 to f1
            )

            if not dual_graph.has_edge(f1_id, f2_id):
                dual_graph.add_edge(f1_id, f2_id, weight=dist, half_edge=he.id)

    start_face_id = view.faces[0].id
    predecessors, _ = nx.dijkstra_predecessor_and_distance(dual_graph, start_face_id)

    spanning_tree = nx.Graph()
    for face_id, predecessor_list in predecessors.items():
        if face_id != start_face_id and predecessor_list:
            spanning_tree.add_edge(
                face_id,
                predecessor_list[0],
                **dual_graph.get_edge_data(face_id, predecessor_list[0])
            )

    return dual_graph, spanning_tree


def compute_fundamental_domain(
    factory: SurfaceFactory, surface_id: SurfaceId, geometry: Any
):
    """
    Constructs a fundamental domain for a given closed, orientable surface.

    This function is geometry-agnostic and relies on the provided 'geometry'
    object to perform geometry-specific calculations (e.g., for Euclidean or
    Hyperbolic space).

    The algorithm proceeds in the following steps:
    1.  Compute local, centered geometric embeddings for each face.
    2.  Compute the isometries (transformations) that align adjacent faces.
    3.  Build a dual graph of the mesh, where nodes are faces and edges
        represent adjacency. Edge weights are the "distance" between face
        centers, computed using the isometries.
    4.  Find a shortest-path spanning tree of the dual graph. This tree
        represents the "canonical" way to lay out the faces without overlap.
    5.  Classify the holonomy of the remaining dual graph edges (the "seams").
        These seams correspond to the generators of the fundamental group.
    6.  Lay out the faces in a global coordinate system by traversing the
        spanning tree and applying the computed isometries.
    7.  Compute the geometric transformations for each holonomy generator by
        composing isometries along paths in the spanning tree.

    Returns:
        A tuple containing all computed artifacts:
        (dual_graph, centered_embeddings, isometries, spanning_tree,
         holonomy_classes, placements, holonomy_transforms, generators)
    """
    view = factory.create_viewer(surface_id, custom_properties={"length": None})

    # 1. Compute local geometry for each face.
    centered_embeddings: Dict[FaceId, Dict[VertexId, np.ndarray]] = {
        face.id: geometry.get_centered_face_embedding(face) for face in view.faces
    }

    # 2. Compute isometries that align adjacent faces.
    isometries: Dict[HalfEdgeId, np.ndarray] = {}
    for he in view.half_edges:
        if he.twin and he.id not in isometries:
            face1_id, face2_id = he.face.id, he.twin.face.id
            v1_id, v2_id = he.origin.id, he.twin.origin.id

            p1, p2 = (
                centered_embeddings[face1_id][v1_id],
                centered_embeddings[face1_id][v2_id],
            )
            q1, q2 = (
                centered_embeddings[face2_id][v1_id],
                centered_embeddings[face2_id][v2_id],
            )

            # Isometry from face2's frame to face1's frame.
            # The edge in face1 is (p1, p2) [from v1 to v2].
            # The corresponding edge in face2 is (q1, q2) [from v1 to v2].
            # Isometry from face2's frame to face1's frame.
            # This should map v1's position in face2 (q1) to v1's position in face1 (p1),
            # and v2's position in face2 (q2) to v2's position in face1 (p2).
            isometry = geometry.get_isometry_aligning_edges(q1, q2, p1, p2)

            isometries[he.id] = isometry
            isometries[he.twin.id] = geometry.invert_transform(isometry)

    # 3 & 4. Build the weighted dual graph and compute the shortest-path spanning tree.
    dual_graph, spanning_tree = _build_weighted_dual_graph_and_spanning_tree(
        view, centered_embeddings, isometries, geometry
    )

    # 5. Classify holonomy based on the computed spanning tree.
    generators, holonomy_classes = _classify_holonomy(view, spanning_tree, dual_graph)

    # 6. Layout the domain by traversing the spanning tree.
    placed_triangles, global_transforms = _layout_domain(
        view, spanning_tree, dual_graph, centered_embeddings, isometries, geometry
    )

    # 7. Compute geometric transforms for holonomy generators.
    holonomy_transforms = {}
    processed_generators = set()

    for he in view.half_edges:
        h_class = holonomy_classes[he.id]
        if h_class and h_class not in processed_generators:
            f1_id, f2_id = he.face.id, he.twin.face.id

            try:
                # Find the path in the spanning tree between the two faces of the seam.
                tree_path = nx.shortest_path(spanning_tree, source=f1_id, target=f2_id)
            except nx.NetworkXNoPath:
                continue

            # Compose isometries along the tree path to get the transform from f2's frame to f1's frame.
            path_transform = geometry.identity_transform()
            current_face_id = f2_id
            for prev_face_id in reversed(tree_path[:-1]):
                path_he_id = dual_graph[current_face_id][prev_face_id]["half_edge"]
                path_he = view.get_proxy(path_he_id)

                # We need the transform from current_face_id's frame to prev_face_id's frame.
                # Let's call this T_current_to_prev.
                # isometries[he.id] is defined as T_he.twin.face -> he.face
                if path_he.face.id == prev_face_id:
                    # path_he.face is prev, path_he.twin.face is current.
                    # We need T_current_to_prev, which is isometries[path_he.id].
                    local_isometry = isometries[path_he.id]
                else:  # path_he.face.id == current_face_id
                    # path_he is in current, path_he.twin.face is prev.
                    # We need T_current_to_prev, which is isometries[path_he.twin.id].
                    local_isometry = isometries[path_he.twin.id]

                # Correct composition order: T_new = T_local @ T_old
                path_transform = local_isometry @ path_transform
                current_face_id = prev_face_id

            # At this point, path_transform is T_f2_to_f1_tree.
            # The holonomy is a closed loop transform. We want to go from f1, across the seam to f2,
            # then back along the tree to f1.
            # T_f1_to_f2_seam is isometries[he.twin.id].
            # T_f2_to_f1_tree is path_transform.
            # The loop transform is T_f2_to_f1_tree @ T_f1_to_f2_seam.
            loop_transform = path_transform @ isometries[he.twin.id]

            # The holonomy generator is the transform that zips up the domain.
            # This is the loop transform, conjugated into the global coordinate system.
            T_f1_to_global = global_transforms[f1_id]
            inv_T_f1_to_global = geometry.invert_transform(T_f1_to_global)
            global_loop_transform = T_f1_to_global @ loop_transform @ inv_T_f1_to_global

            holonomy_transforms[h_class] = global_loop_transform
            processed_generators.add(h_class)

            inv_class = _get_holonomy_inverse(h_class)
            if inv_class and inv_class not in processed_generators:
                # The transform for the inverse generator is the inverse of the loop transform.
                holonomy_transforms[inv_class] = geometry.invert_transform(
                    global_loop_transform
                )
                processed_generators.add(inv_class)

    return (
        dual_graph,
        centered_embeddings,
        isometries,
        spanning_tree,
        holonomy_classes,
        placed_triangles,
        holonomy_transforms,
        generators,
    )
