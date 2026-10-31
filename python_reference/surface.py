"""
Implements the core data structures for representing a 2D manifold mesh
and exposes utility functions from `mazedemo.surface_util`.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union, Hashable
import collections

from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId

# --- THE "FACADE" IMPORT ---
# This makes all functions from surface_util available through the surface namespace,
# ensuring backward compatibility for all other modules.
from mazedemo.surface_util import *


# --- Forward Declarations for Type Hinting ---
class Surface:
    pass


class Vertex:
    pass


class Face:
    pass


class HalfEdge:
    pass


# --- Immutable Surface Representation ---


@dataclass(frozen=True)
class FrozenSurface:
    """An immutable, purely topological, data-centric representation of a mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    half_edges: np.ndarray

    def to_workbench(self, surface_id: SurfaceId) -> "Surface":
        """Creates a mutable, object-graph view (Surface) of this surface."""
        workbench = Surface()
        num_verts, num_faces, num_hes = (
            self.vertices.shape[0],
            self.faces.shape[0],
            self.half_edges.shape[0],
        )

        workbench.vertices = [
            workbench.new_vertex(parent_ids=(VertexId(surface_id, i),))
            for i in range(num_verts)
        ]
        workbench.faces = [
            workbench.new_face(parent_ids=(FaceId(surface_id, i),))
            for i in range(num_faces)
        ]

        temp_hes = [None] * num_hes
        for i in range(num_hes):
            origin_v_idx = self.half_edges[i, 0]
            if origin_v_idx != -1:
                origin_v = workbench.vertices[origin_v_idx]
                he = workbench.new_half_edge(
                    origin=origin_v, parent_ids=(HalfEdgeId(surface_id, i),)
                )
                temp_hes[i] = he
        workbench.half_edges = [he for he in temp_hes if he is not None]

        for i, he_data in enumerate(self.half_edges):
            he = temp_hes[i]
            if he is None:
                continue
            twin_idx, next_idx, prev_idx, face_idx = (
                he_data[1],
                he_data[2],
                he_data[3],
                he_data[4],
            )
            if twin_idx != -1:
                he.twin = temp_hes[twin_idx]
            if next_idx != -1:
                he.next = temp_hes[next_idx]
            if prev_idx != -1:
                he.prev = temp_hes[prev_idx]
            if face_idx != -1:
                he.face = workbench.faces[face_idx]

        for i, v_data in enumerate(self.vertices):
            he_idx = v_data[0]
            if he_idx != -1:
                workbench.vertices[i].half_edge = temp_hes[he_idx]

        for i, f_data in enumerate(self.faces):
            he_idx = f_data[0]
            if he_idx != -1:
                workbench.faces[i].half_edge = temp_hes[he_idx]

        return workbench


# --- Mutable Surface Data Structures ---


@dataclass
class Vertex:
    id: int
    parent_ids: Tuple[VertexId, ...]
    half_edge: Optional["HalfEdge"] = None
    is_deleted: bool = field(default=False, init=False)

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.id)

    @property
    def half_edges(self) -> List["HalfEdge"]:
        # Delegated to shared utility
        return collect_outgoing_half_edges(self)


@dataclass
class Face:
    id: int
    parent_ids: Tuple[FaceId, ...]
    half_edge: Optional["HalfEdge"] = None
    is_deleted: bool = field(default=False, init=False)

    def __eq__(self, other):
        return isinstance(other, Face) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"{self.id}"

    @property
    def half_edges(self) -> List["HalfEdge"]:
        edges = []
        current_he = self.half_edge
        for _ in range(1000):  # Safety break for malformed loops
            edges.append(current_he)
            current_he = current_he.next
            if current_he == self.half_edge:
                return edges
        # If the loop finishes without breaking, the topology is broken.
        return []

    @property
    def vertices(self) -> List[Vertex]:
        return [he.origin for he in self.half_edges]


@dataclass(eq=False)
class HalfEdge:
    id: int
    origin: Vertex
    parent_ids: Tuple[HalfEdgeId, ...]
    twin: Optional["HalfEdge"] = None
    face: Optional[Face] = None
    next: Optional["HalfEdge"] = None
    prev: Optional["HalfEdge"] = None
    is_deleted: bool = field(default=False, init=False)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    def __str__(self):
        return str(self.id)


# --- The Workbench Class ---


@dataclass
class Surface:
    """A mutable, object-graph representation of a manifold mesh (a workbench)."""

    vertices: List[Vertex] = field(default_factory=list)
    half_edges: List[HalfEdge] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    source_surface_ids: Optional[Set[SurfaceId]] = field(init=False, default=None)
    _v_counter: int = field(init=False, default=0)
    _f_counter: int = field(init=False, default=0)
    _parent_id_map: Dict[ElementId, List[Any]] = field(
        init=False, default_factory=lambda: collections.defaultdict(list)
    )
    traced_descendants: Optional[Dict[ElementId, Set[Any]]] = field(
        init=False, default=None
    )

    def __post_init__(self):
        if self.vertices:
            valid_v_ids = [v.id for v in self.vertices if v is not None]
            if valid_v_ids:
                self._v_counter = max(valid_v_ids) + 1
        if self.faces:
            valid_f_ids = [f.id for f in self.faces if f is not None]
            if valid_f_ids:
                self._f_counter = max(valid_f_ids) + 1
        self._build_parent_index()

    def _build_parent_index(self):
        self._parent_id_map.clear()
        for elem_list in [self.vertices, self.faces, self.half_edges]:
            for elem in elem_list:
                if elem:
                    self._add_to_parent_index(elem)

    def _add_to_parent_index(self, element: Any):
        for parent_id in element.parent_ids:
            self._parent_id_map[parent_id].append(element)

    def _remove_from_parent_index(self, element: Any):
        """Incrementally removes an element from the parent index."""
        for parent_id in element.parent_ids:
            if parent_id in self._parent_id_map:
                try:
                    self._parent_id_map[parent_id].remove(element)
                    if not self._parent_id_map[parent_id]:
                        del self._parent_id_map[parent_id]
                except ValueError:
                    # Element was already removed, which is fine.
                    pass

    def get_descendants(self, ancestor_id: ElementId) -> List[Any]:
        """
        Finds all elements in this workbench that are descendants of the given
        ancestor ID.
        """
        if self.traced_descendants is not None:
            if ancestor_id in self.traced_descendants:
                return list(self.traced_descendants.get(ancestor_id, set()))

        if ancestor_id in self._parent_id_map:
            return self._parent_id_map[ancestor_id]

        if (
            self.source_surface_ids is not None
            and ancestor_id.surface_id not in self.source_surface_ids
        ):
            raise ValueError(
                f"Ancestor {ancestor_id} is from a surface that is not a source for this "
                "workbench, and no trace was performed."
            )

        return []

    def get_descendant(self, ancestor_id: ElementId) -> Optional[Any]:
        """
        Finds the unique element in this workbench that is a descendant of the
        given ancestor ID. Raises an error if zero or multiple descendants are found.
        """
        descendants = self.get_descendants(ancestor_id)
        if len(descendants) > 1:
            raise ValueError(f"Found multiple descendants for ancestor {ancestor_id}")
        if not descendants:
            raise ValueError(f"Found zero descendants for ancestor {ancestor_id}")
        return descendants[0]

    def new_vertex(self, parent_ids: Tuple[VertexId, ...] = ()) -> Vertex:
        v = Vertex(id=self._v_counter, parent_ids=parent_ids)
        self._v_counter += 1
        self.vertices.append(v)
        self._add_to_parent_index(v)
        return v

    def new_face(self, parent_ids: Tuple[FaceId, ...] = ()) -> Face:
        f = Face(id=self._f_counter, parent_ids=parent_ids)
        self._f_counter += 1
        self.faces.append(f)
        self._add_to_parent_index(f)
        return f

    def new_half_edge(
        self, origin: Vertex, parent_ids: Tuple[HalfEdgeId, ...] = ()
    ) -> HalfEdge:
        he = HalfEdge(id=len(self.half_edges), origin=origin, parent_ids=parent_ids)
        self.half_edges.append(he)
        self._add_to_parent_index(he)
        return he

    def delete_vertex(self, vertex: Vertex):
        vertex.is_deleted = True
        self._remove_from_parent_index(vertex)  # FIX: Update parent index

    def delete_face(self, face: Face):
        face.is_deleted = True
        self._remove_from_parent_index(face)  # FIX: Update parent index

    def delete_half_edge(self, half_edge: HalfEdge):
        half_edge.is_deleted = True
        self._remove_from_parent_index(half_edge)  # FIX: Update parent index

    def to_frozen(self) -> "FrozenSurface":
        """
        Converts the mutable workbench representation into a compact, immutable,
        array-based FrozenSurface, correctly filtering out any elements that
        have been marked for deletion.
        """
        # Get all elements that haven't been marked for deletion
        live_vertices = [
            v for v in self.vertices if not getattr(v, "is_deleted", False)
        ]
        live_faces = [f for f in self.faces if not getattr(f, "is_deleted", False)]
        live_half_edges = [
            he for he in self.half_edges if not getattr(he, "is_deleted", False)
        ]

        # Create a mapping from the temporary Python object ID to its final index in the array.
        vert_map = {id(v): i for i, v in enumerate(live_vertices)}
        face_map = {id(f): i for i, f in enumerate(live_faces)}
        he_map = {id(he): i for i, he in enumerate(live_half_edges)}

        # Build the half-edge data array.
        if not live_half_edges:
            he_array = np.empty((0, 5), dtype=int)
        else:
            he_array = np.array(
                [
                    [
                        vert_map.get(id(he.origin), -1),
                        he_map.get(id(he.twin), -1) if he.twin else -1,
                        he_map.get(id(he.next), -1) if he.next else -1,
                        he_map.get(id(he.prev), -1) if he.prev else -1,
                        face_map.get(id(he.face), -1) if he.face else -1,
                    ]
                    for he in live_half_edges
                ],
                dtype=int,
            )

        # Build the vertex data array.
        if not live_vertices:
            vert_array = np.empty((0, 1), dtype=int)
        else:
            vert_array = np.array(
                [
                    he_map.get(id(v.half_edge), -1) if v.half_edge else -1
                    for v in live_vertices
                ],
                dtype=int,
            ).reshape(-1, 1)

        # Build the face data array.
        if not live_faces:
            face_array = np.empty((0, 1), dtype=int)
        else:
            face_array = np.array(
                [
                    he_map.get(id(f.half_edge), -1) if f.half_edge else -1
                    for f in live_faces
                ],
                dtype=int,
            ).reshape(-1, 1)

        return FrozenSurface(vertices=vert_array, faces=face_array, half_edges=he_array)

    def populate_from_faces(self, faces: List[List[int]]):
        """
        Populates the surface's topology from a list of faces.

        This method builds the half-edge data structure from a list of vertex
        indices for each face. It automatically creates the required Vertex, Face,
        and HalfEdge objects and correctly identifies and links twin half-edges.
        Boundary edges (those belonging to only one face) will correctly have
        no twin.
        """
        from mazedemo.surface_util import is_oriented, orient_faces

        if self.vertices or self.faces or self.half_edges:
            pass  # Populating a non-empty surface is allowed

        oriented_faces = orient_faces(faces) if not is_oriented(faces) else faces
        # Ensure deterministic ordering of vertices to match grid indices
        all_vert_indices = sorted(list(set(v for f in oriented_faces for v in f)))
        if not all_vert_indices:
            return

        temp_vertices = {i: self.new_vertex() for i in all_vert_indices}

        edge_to_he_map: Dict[Tuple[int, int], HalfEdge] = {}

        for poly_indices in oriented_faces:
            face = self.new_face()
            poly_hes = [
                self.new_half_edge(origin=temp_vertices[v_idx])
                for v_idx in poly_indices
            ]

            for j, he in enumerate(poly_hes):
                he.face = face
                he.next = poly_hes[(j + 1) % len(poly_hes)]
                he.prev = poly_hes[(j - 1 + len(poly_hes)) % len(poly_hes)]
                if not temp_vertices[poly_indices[j]].half_edge:
                    temp_vertices[poly_indices[j]].half_edge = he
            face.half_edge = poly_hes[0]

            for j, he in enumerate(poly_hes):
                u_id = poly_indices[j]
                v_id = poly_indices[(j + 1) % len(poly_hes)]

                edge_key = tuple(sorted((u_id, v_id)))

                if edge_key in edge_to_he_map:
                    twin_he = edge_to_he_map.pop(edge_key)
                    he.twin, twin_he.twin = twin_he, he
                else:
                    edge_to_he_map[edge_key] = he

        self.__post_init__()

    def __str__(self) -> str:
        return get_topology_str(self)

    def is_valid(
        self, allow_missing_twin=False, allow_missing_face=False, allow_multi_edge=True
    ) -> bool:
        """Performs validation by calling the shared utility function."""
        return validate_topology(
            self, allow_missing_twin, allow_missing_face, allow_multi_edge
        )

    def chi(self) -> int:
        """Calculates the Euler characteristic."""
        return calculate_chi(self)
