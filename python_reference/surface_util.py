"""
Provides standalone utility and validation functions that operate on Surface
or Viewer objects, as defined in `mazedemo.surface`.
"""

from typing import List, Union, TYPE_CHECKING, Tuple, Dict
import collections
import copy
import networkx as nx
from absl import logging

# Use a TYPE_CHECKING block to import for type hints, avoiding circular dependencies at runtime.
if TYPE_CHECKING:
    from mazedemo.surface import Surface
    from mazedemo.viewer import Viewer

# --- PUBLIC HELPER FUNCTIONS ---


def has_missing_twin(surface_or_view: Union["Surface", "Viewer"]) -> bool:
    """Checks if any half-edge in the surface is missing a twin."""
    return any(he.twin is None for he in surface_or_view.half_edges)


def has_missing_face(surface_or_view: Union["Surface", "Viewer"]) -> bool:
    """Checks if any half-edge in the surface belongs to a boundary loop."""
    return any(he.face is None for he in surface_or_view.half_edges)


def has_parallel_edges(surface_or_view: Union["Surface", "Viewer"]) -> bool:
    """
    Checks if the surface has any parallel edges (multiple edges connecting the
    same two vertices).
    """
    for v in surface_or_view.vertices:
        # This check is robust for deleted vertices as they won't be in the list.
        neighbors = []
        for he in collect_outgoing_half_edges(v):
            if he.twin:
                neighbors.append(he.twin.origin.id)
        if len(neighbors) != len(set(neighbors)):
            return True
    return False


# --- TRAVERSAL UTILITIES ---


def collect_outgoing_half_edges(vertex) -> List:
    """
    Returns all half-edges originating from a vertex, handling boundaries robustly.

    This generic function works on any object (Surface Vertex, Viewer Proxy, etc.)
    that follows the half-edge interface:
      - vertex.half_edge -> HalfEdge
      - half_edge.twin -> HalfEdge (or None)
      - half_edge.next -> HalfEdge
      - half_edge.prev -> HalfEdge
    """
    start_he = vertex.half_edge
    if not start_he:
        return []

    # 1. Swing Clockwise (CW) to find the start of the fan (or loop back).
    # CW neighbor of 'he' is 'he.twin.next'.
    curr = start_he
    for _ in range(1000):  # Safety limit
        if curr.twin is None:
            # We hit a boundary edge while swinging CW. This is the start of the fan.
            break
        next_cw = curr.twin.next
        if next_cw is None:  # Should not happen in valid mesh but robust check
            break
        if next_cw == start_he:
            # We circled back to the start; it's an interior vertex.
            break
        curr = next_cw

    start_of_fan = curr

    # 2. Swing Counter-Clockwise (CCW) to collect all edges.
    # CCW neighbor of 'he' is 'he.prev.twin'.
    outgoing_hes = []
    curr = start_of_fan
    for _ in range(1000):
        outgoing_hes.append(curr)

        # Move CCW
        if curr.prev is None:
            break
        next_ccw = curr.prev.twin

        if next_ccw is None:
            # Hit the other boundary. Done.
            break

        curr = next_ccw
        if curr == start_of_fan:
            # Full circle completed.
            break

    return outgoing_hes


# --- VALIDATION ---


def validate_topology(
    surface_or_view: Union["Surface", "Viewer"],
    allow_missing_twin=False,
    allow_missing_face=False,
    allow_multi_edge=True,
) -> bool:
    """Performs a comprehensive validation of topological integrity."""
    is_ok = True
    try:
        half_edges = surface_or_view.half_edges
        vertices = surface_or_view.vertices
        faces = surface_or_view.faces
    except AttributeError:
        logging.error(
            "validate_topology fail: Input object does not have a valid surface interface."
        )
        return False

    for he in half_edges:
        if not he.next or he.next.prev != he:
            logging.error("validate_topology fail: he.next.prev != he")
            is_ok = False

    if not allow_missing_twin and has_missing_twin(surface_or_view):
        for he in half_edges:
            if he.twin is None:
                logging.error(
                    "validate_topology fail: Half-edge %s has no twin, but allow_missing_twin=False.",
                    repr(getattr(he, "id", "N/A")),
                )
        is_ok = False

    if not allow_missing_face and has_missing_face(surface_or_view):
        for he in half_edges:
            if he.face is None:
                logging.error(
                    "validate_topology fail: Half-edge %s has no face, but allow_missing_face=False.",
                    repr(getattr(he, "id", "N/A")),
                )
        is_ok = False

    for he in half_edges:
        if getattr(he, "is_deleted", False):
            continue
        if he.twin:
            if getattr(he.twin, "is_deleted", False):
                continue
            if he.twin.twin != he:
                is_ok = False
                logging.error(
                    f"validate_topology fail: Twin consistency broken for {he.id}."
                )
            if he.origin == he.twin.origin:
                is_ok = False
                logging.error(
                    f"validate_topology fail: Self-loop detected for {he.id}."
                )
        elif not allow_missing_twin:
            is_ok = False

    for v in vertices:
        if getattr(v, "is_deleted", False):
            continue
        if not v.half_edge:
            is_origin = any(
                he.origin
                and he.origin.id == v.id
                and not getattr(he, "is_deleted", False)
                for he in half_edges
            )
            if is_origin:
                is_ok = False
                logging.error(
                    "validate_topology fail: Vertex %s is an origin but has no half_edge pointer.",
                    repr(v.id),
                )
            continue
        if v.half_edge.origin.id != v.id:
            is_ok = False
            logging.error(
                "validate_topology fail: Vertex %s points to a half_edge %s that does not originate from it.",
                repr(v.id),
                repr(v.half_edge.id),
            )

    for f in faces:
        if getattr(f, "is_deleted", False):
            continue
        if not f.half_edge:
            is_ok = False
            logging.error(
                "validate_topology fail: Face %s has no half_edge pointer.", repr(f.id)
            )
            continue
        start_he = f.half_edge
        current_he = start_he
        for _ in range(1000):  # Safety break
            if current_he is None or getattr(current_he, "is_deleted", False):
                is_ok = False
                break
            if current_he.face is None or current_he.face.id != f.id:
                is_ok = False
                break
            current_he = current_he.next
            if current_he == start_he:
                break
        else:
            is_ok = False
        if not is_ok:
            logging.error(
                "validate_topology fail: Face %s does not form a valid cycle.",
                repr(f.id),
            )

    if not allow_multi_edge:
        for v in vertices:
            if getattr(v, "is_deleted", False):
                continue

            neighbors = []
            for he in collect_outgoing_half_edges(v):
                if he.twin:
                    neighbors.append(he.twin.origin.id)

            neighbor_counts = collections.Counter(neighbors)
            for neighbor_id, count in neighbor_counts.items():
                if count > 1:
                    is_ok = False
                    logging.error(
                        "validate_topology fail: Parallel edges detected between vertex %s and %s (count=%d), but allow_multi_edge=False.",
                        v.id,
                        neighbor_id,
                        count,
                    )

    actual_outgoing = collections.defaultdict(set)
    for he in surface_or_view.half_edges:
        if not getattr(he, "is_deleted", False) and he.origin:
            actual_outgoing[he.origin.id].add(he.id)

    for v in surface_or_view.vertices:
        if getattr(v, "is_deleted", False):
            continue

        # Use the unified utility to get reachable edges
        reachable_edges = {he.id for he in collect_outgoing_half_edges(v)}
        all_edges = actual_outgoing[v.id]

        if reachable_edges != all_edges:
            is_ok = False
            logging.error(
                "validate_topology fail: Vertex %s is non-manifold (disconnected fans). "
                "Reachable: %s, Actual: %s",
                v.id,
                reachable_edges,
                all_edges,
            )

    if not is_ok:
        logging.error("Surface failed validation.")
    return is_ok


# --- ORIENTATION AND ANALYSIS UTILITIES ---


def is_oriented(faces: List[List[int]]) -> bool:
    """
    Checks if the faces of a mesh have a consistent topological orientation.

    NOTE: This implementation is known to be fragile. It can be fooled by
    certain non-manifold topologies where an edge is shared by more than two
    faces, but the directed edges happen to not conflict. A more robust
    implementation would check that the count of each undirected edge is <= 2.
    """
    directed_edges = []
    for face in faces:
        for i in range(len(face)):
            u, v = face[i], face[(i + 1) % len(face)]
            directed_edges.append((u, v))
    return len(directed_edges) == len(set(directed_edges))


def orient_faces(faces: List[List[int]]) -> List[List[int]]:
    """
    Attempts to orient all faces in a mesh consistently using a traversal
    of the face adjacency graph's minimum spanning tree.

    NOTE: This implementation is known to be fragile. Its logic for finding
    the shared edge between two faces is inefficient and can fail on complex
    meshes. It also does not correctly handle meshes with multiple disconnected
    components in their dual graph.
    """
    if not faces:
        return []
    edge_to_faces = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(len(face)):
            u, v = face[j], face[(j + 1) % len(face)]
            edge_to_faces[tuple(sorted((u, v)))].append(i)

    dual_graph = nx.Graph()
    for _, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:
            dual_graph.add_edge(face_indices[0], face_indices[1])

    if not list(dual_graph.nodes()):
        return faces

    mst = nx.minimum_spanning_tree(dual_graph)
    oriented_faces = copy.deepcopy(faces)
    root = list(mst.nodes)[0]

    for u, v in nx.bfs_edges(mst, source=root):
        curr_face = oriented_faces[u]
        neighbor_face = oriented_faces[v]

        shared_edge_key = None
        for i in range(len(curr_face)):
            key = tuple(sorted((curr_face[i], curr_face[(i + 1) % len(curr_face)])))
            if v in edge_to_faces.get(key, []):
                shared_edge_key = key
                break

        edge_in_curr = None
        for i in range(len(curr_face)):
            if (
                tuple(sorted((curr_face[i], curr_face[(i + 1) % len(curr_face)])))
                == shared_edge_key
            ):
                edge_in_curr = (curr_face[i], curr_face[(i + 1) % len(curr_face)])
                break

        if not edge_in_curr:
            continue

        is_reversed = any(
            (neighbor_face[i], neighbor_face[(i + 1) % len(neighbor_face)])
            == (edge_in_curr[1], edge_in_curr[0])
            for i in range(len(neighbor_face))
        )
        if not is_reversed:
            oriented_faces[v].reverse()

    if not is_oriented(oriented_faces):
        raise ValueError("Mesh is not orientable (e.g., a Möbius strip).")

    return oriented_faces


def calculate_chi(surface_or_view: Union["Surface", "Viewer"]) -> int:
    """
    Calculates the Euler characteristic (V - E + F) for a surface.

    This pure function operates on any object that provides the standard
    surface interface (`vertices`, `faces`, `half_edges` properties).

    Args:
        surface_or_view: The Surface workbench or Viewer to analyze.

    Returns:
        The integer Euler characteristic of the surface.
    """
    # No type-switching or None-filtering is needed if we assume a clean interface.
    num_vertices = len(surface_or_view.vertices)
    num_faces = len(surface_or_view.faces)
    half_edges = surface_or_view.half_edges

    num_missing_twins = len([he for he in half_edges if he.twin is None])

    # Each edge corresponds to two half-edges in a sealed manifold.
    # Boundary edges have no twin, so we count them once.
    num_edges = (len(half_edges) + num_missing_twins) // 2

    return num_vertices - num_edges + num_faces


def get_topology_str(
    surface_or_view: Union["Surface", "Viewer"], properties: List[str] = None
) -> str:
    """
    Generates a detailed, human-readable string representation of the object's
    topology, optionally including specified properties.
    """
    properties = properties or []
    try:
        s = f"{surface_or_view.__class__.__name__}(V={len(surface_or_view.vertices)}, F={len(surface_or_view.faces)})\n"
        s += "--- Vertices (id: out_he) ---\n"
        for v in surface_or_view.vertices:
            s += f"  {v.id}: he={getattr(v.half_edge, 'id', 'N')}, num_parents={len(v.parent_ids)}\n"

        # Build header for half-edge properties
        s += f"--- Half-Edges  ---\n"

        for he in surface_or_view.half_edges:
            line = (
                f"  {he.id}: o={getattr(he.origin, 'id', '?')} ->"
                f"{getattr(he.next.origin, 'id', 'N')}, "
                f"t={getattr(he.twin, 'id', 'N')}, "
                f"n={getattr(he.next, 'id', 'N')}, "
                f"p={getattr(he.prev, 'id', 'N')}, "
                f"f={getattr(he.face, 'id', 'N')}, "
                f"num_parents={len(he.parent_ids)}"
            )

            for prop_name in properties:
                try:
                    value = getattr(he, prop_name)
                    if prop_name == "parent_ids":
                        # Special case for parent_ids: show count instead of full list
                        val_repr = str(len(value)) if value is not None else "0"
                    elif isinstance(value, float):
                        val_repr = f"{value:.4f}"
                    elif value is None:
                        val_repr = "None"
                    else:
                        val_repr = str(value)
                    line += f", {prop_name}={val_repr}"
                except AttributeError:
                    line += f", {prop_name}=N/A"

            s += line + "\n"

        return s
    except Exception as e:
        import traceback

        logging.error(
            "Failed to generate topology repr: %s\n%s", e, traceback.format_exc()
        )
        return f"Could not generate repr due to corruption: {e}"
