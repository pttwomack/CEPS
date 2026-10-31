# mazedemo/viewer.py
"""
Defines the Viewer class and its associated read-only proxy classes, which
provide the primary API for inspecting a surface and its properties.
"""

from typing import Any, Dict, Hashable, List, Set, Tuple, Type, Optional
from functools import lru_cache
import re
from absl import logging

from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId
from mazedemo.surface import FrozenSurface, get_topology_str
from mazedemo.surface_util import collect_outgoing_half_edges


# Forward declarations to avoid circular imports
class SurfaceFactory:
    pass


class SurfacePropertyManager:
    pass


# --- Read-Only Proxy Classes ---


class _ElementProxy:
    """Base class for read-only element proxies."""

    def __init__(self, viewer: "Viewer", element_id: ElementId):
        self._viewer = viewer
        self.id = element_id

    @property
    def parent_ids(self) -> Tuple[ElementId, ...]:
        """The parent IDs of this element from the previous surface version(s)."""
        return self._viewer.get_parents(self.id)

    def get_property(self, prop_name: str, default: Any = None) -> Any:
        """Gets a property for this element, searching through history."""
        return self._viewer.get_property(self.id, prop_name, default)

    def __getattr__(self, name: str) -> Any:
        """Convenience method to access properties directly."""
        prop = self.get_property(name)
        if prop is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute or property '{name}'"
            )
        return prop


class Vertex(_ElementProxy):
    """A read-only proxy for a vertex."""

    @property
    def half_edge(self) -> Optional["HalfEdge"]:
        """An arbitrary half-edge originating from this vertex."""
        return self._viewer.get_vertex_half_edge(self.id)

    @property
    def half_edges(self) -> List["HalfEdge"]:
        """Returns all half-edges originating from this vertex, handling boundaries correctly."""
        start_he = self.half_edge
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
            if next_cw.id == start_he.id:
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
                break  # Invalid topology
            next_ccw = curr.prev.twin

            if next_ccw is None:
                # Hit the other boundary. Done.
                break

            curr = next_ccw
            if curr.id == start_of_fan.id:
                # Full circle completed.
                break

        return outgoing_hes

    @property
    def faces(self) -> List["Face"]:
        """Returns all faces adjacent to this vertex."""
        return self._viewer.get_vertex_faces(self.id)


class Face(_ElementProxy):
    """A read-only proxy for a face."""

    @property
    def half_edge(self) -> Optional["HalfEdge"]:
        """An arbitrary half-edge on the boundary of this face."""
        return self._viewer.get_face_entry_half_edge(self.id)

    @property
    def half_edges(self) -> List["HalfEdge"]:
        """The cycle of half-edges bounding this face."""
        return self._viewer.get_face_half_edges(self.id)

    @property
    def vertices(self) -> List["Vertex"]:
        """Returns all vertices belonging to this face."""
        return [he.origin for he in self.half_edges]


class HalfEdge(_ElementProxy):
    """A read-only proxy for a half-edge."""

    @property
    def origin(self) -> Optional[Vertex]:
        """The origin vertex of this half-edge."""
        return self._viewer.get_origin(self.id)

    @property
    def twin(self) -> Optional["HalfEdge"]:
        """The twin half-edge."""
        return self._viewer.get_twin(self.id)

    @property
    def next(self) -> Optional["HalfEdge"]:
        """The next half-edge in the face cycle."""
        return self._viewer.get_next(self.id)

    @property
    def prev(self) -> Optional["HalfEdge"]:
        """The previous half-edge in the face cycle."""
        return self._viewer.get_prev(self.id)

    @property
    def face(self) -> Optional[Face]:
        """The face this half-edge belongs to."""
        return self._viewer.get_face(self.id)


_ELEMENT_TYPE_STRING_CACHE = {}


# --- The Viewer Class ---


class Viewer:
    """
    A read-only, user-facing handle to a specific version of a surface.
    """

    def __init__(
        self,
        factory: "SurfaceFactory",
        surface_id: SurfaceId,
        custom_properties: Dict[str, Any] = None,
    ):
        self._factory = factory
        self._surface_id = surface_id

        if surface_id not in self._factory._commits:
            raise ValueError(
                f"Viewer error: No commit found for SurfaceId {surface_id}."
            )

        self._commit = self._factory._commits[surface_id]
        topology_id = self._commit.topology_id

        if topology_id not in self._factory._topologies:
            raise ValueError(
                f"Viewer error for SurfaceId {surface_id}: The associated TopologyId {topology_id} "
                f"was not found in the factory's registered topologies."
            )

        self._frozen: FrozenSurface = self._factory._topologies.get(topology_id)

        if self._frozen is None:
            raise ValueError(
                f"Viewer error for SurfaceId {surface_id}: Topology {topology_id} is present but resolves to None."
            )

        self._prop_manager: SurfacePropertyManager = self._factory._prop_manager
        self._proxy_cache: Dict[ElementId, _ElementProxy] = {}

        self._proxy_classes = {"vertex": Vertex, "face": Face, "half_edge": HalfEdge}

        if custom_properties:
            prop_key = tuple(sorted(custom_properties.items()))
            if prop_key in self._factory._proxy_class_cache:
                self._proxy_classes = self._factory._proxy_class_cache[prop_key]
            else:
                # Dynamically create new proxy classes with the custom properties.
                new_proxy_classes = {
                    "vertex": type(
                        f"VertexProxy_{id(custom_properties)}", (Vertex,), {}
                    ),
                    "face": type(f"FaceProxy_{id(custom_properties)}", (Face,), {}),
                    "half_edge": type(
                        f"HalfEdgeProxy_{id(custom_properties)}", (HalfEdge,), {}
                    ),
                }

                for new_prop, spec in custom_properties.items():

                    def make_prop_getter(prop_name):
                        def prop_getter(self):
                            value = self.get_property(prop_name)
                            if value is None:
                                raise AttributeError(
                                    f"'{self.__class__.__name__}' object has no attribute "
                                    f"or property '{prop_name}'"
                                )
                            return value

                        return prop_getter

                    if spec is None:
                        prop_getter_func = make_prop_getter(new_prop)
                    elif isinstance(spec, tuple) and len(spec) == 2:
                        prop_name, start_surface_id = spec

                        def make_special_getter(p_name, s_id):
                            def prop_getter(proxy_self):
                                # Calls the new iterative helper on the viewer.
                                value = proxy_self._viewer._find_property_iterative(
                                    s_id, proxy_self.id, p_name
                                )
                                if value is None:
                                    raise AttributeError(
                                        f"'{proxy_self.__class__.__name__}' object has no attribute "
                                        f"or property '{p_name}'"
                                    )
                                return value

                            return prop_getter

                        prop_getter_func = make_special_getter(
                            prop_name, start_surface_id
                        )
                    else:
                        raise ValueError(
                            f"Invalid custom property specification: {spec}"
                        )

                    for proxy_class in new_proxy_classes.values():
                        setattr(proxy_class, new_prop, property(prop_getter_func))

                self._factory._proxy_class_cache[prop_key] = new_proxy_classes
                self._proxy_classes = new_proxy_classes

    @property
    def topology_id(self) -> Hashable:
        """The opaque, unique ID of the topology for this surface version."""
        return self._factory.get_topology_id(self._surface_id)

    def __repr__(self):
        return get_topology_str(self)

    def get_proxy(self, element_id: ElementId) -> Optional[_ElementProxy]:
        """Gets or creates a proxy object for a given ElementId."""
        if element_id is None:
            return None
        if element_id in self._proxy_cache:
            return self._proxy_cache[element_id]

        proxy_class = None
        if isinstance(element_id, VertexId):
            proxy_class = self._proxy_classes["vertex"]
        elif isinstance(element_id, FaceId):
            proxy_class = self._proxy_classes["face"]
        elif isinstance(element_id, HalfEdgeId):
            proxy_class = self._proxy_classes["half_edge"]

        if proxy_class:
            proxy = proxy_class(self, element_id)
            self._proxy_cache[element_id] = proxy
            return proxy
        return None

    # --- Element Accessors ---

    @property
    @lru_cache(maxsize=None)
    def vertex_ids(self) -> List[VertexId]:
        """Returns a list of all persistent VertexIds for this surface."""
        return [
            VertexId(self._surface_id, i) for i in range(self._frozen.vertices.shape[0])
        ]

    @property
    @lru_cache(maxsize=None)
    def face_ids(self) -> List[FaceId]:
        """Returns a list of all persistent FaceIds for this surface."""
        return [FaceId(self._surface_id, i) for i in range(self._frozen.faces.shape[0])]

    @property
    @lru_cache(maxsize=None)
    def half_edge_ids(self) -> List[HalfEdgeId]:
        """Returns a list of all persistent HalfEdgeIds for this surface."""
        return [
            HalfEdgeId(self._surface_id, i)
            for i in range(self._frozen.half_edges.shape[0])
        ]

    @property
    @lru_cache(maxsize=None)
    def vertices(self) -> List[Vertex]:
        """Returns a list of all vertex proxies for this surface."""
        return [self.get_proxy(vid) for vid in self.vertex_ids]

    @property
    @lru_cache(maxsize=None)
    def faces(self) -> List[Face]:
        """Returns a list of all face proxies for this surface."""
        return [self.get_proxy(fid) for fid in self.face_ids]

    @property
    @lru_cache(maxsize=None)
    def half_edges(self) -> List[HalfEdge]:
        """Returns a list of all half-edge proxies for this surface."""
        return [self.get_proxy(hid) for hid in self.half_edge_ids]

    # --- Topological Traversal API (used by proxies) ---

    @lru_cache(maxsize=None)
    def get_origin(self, he_id: HalfEdgeId) -> Optional[Vertex]:
        origin_idx = self._frozen.half_edges[he_id._local_id, 0]
        return (
            self.get_proxy(VertexId(self._surface_id, origin_idx))
            if origin_idx != -1
            else None
        )

    @lru_cache(maxsize=None)
    def get_twin(self, he_id: HalfEdgeId) -> Optional[HalfEdge]:
        twin_idx = self._frozen.half_edges[he_id._local_id, 1]
        return (
            self.get_proxy(HalfEdgeId(self._surface_id, twin_idx))
            if twin_idx != -1
            else None
        )

    @lru_cache(maxsize=None)
    def get_next(self, he_id: HalfEdgeId) -> Optional[HalfEdge]:
        next_idx = self._frozen.half_edges[he_id._local_id, 2]
        return (
            self.get_proxy(HalfEdgeId(self._surface_id, next_idx))
            if next_idx != -1
            else None
        )

    @lru_cache(maxsize=None)
    def get_prev(self, he_id: HalfEdgeId) -> Optional[HalfEdge]:
        prev_idx = self._frozen.half_edges[he_id._local_id, 3]
        return (
            self.get_proxy(HalfEdgeId(self._surface_id, prev_idx))
            if prev_idx != -1
            else None
        )

    @lru_cache(maxsize=None)
    def get_face(self, he_id: HalfEdgeId) -> Optional[Face]:
        face_idx = self._frozen.half_edges[he_id._local_id, 4]
        return (
            self.get_proxy(FaceId(self._surface_id, face_idx))
            if face_idx != -1
            else None
        )

    def get_vertex_half_edge(self, vertex_id: VertexId) -> Optional[HalfEdge]:
        he_idx = self._frozen.vertices[vertex_id._local_id, 0]
        return (
            self.get_proxy(HalfEdgeId(self._surface_id, he_idx))
            if he_idx != -1
            else None
        )

    def get_face_entry_half_edge(self, face_id: FaceId) -> Optional[HalfEdge]:
        he_idx = self._frozen.faces[face_id._local_id, 0]
        return (
            self.get_proxy(HalfEdgeId(self._surface_id, he_idx))
            if he_idx != -1
            else None
        )

    def get_face_half_edges(self, face_id: FaceId) -> List[HalfEdge]:
        start_he_idx = self._frozen.faces[face_id._local_id, 0]
        if start_he_idx == -1:
            return []

        cycle_ids = []
        current_he_idx = start_he_idx
        for _ in range(1000):  # Safety break
            cycle_ids.append(HalfEdgeId(self._surface_id, current_he_idx))
            current_he_idx = self._frozen.half_edges[current_he_idx, 2]  # next
            if current_he_idx == start_he_idx:
                break
        return [self.get_proxy(hid) for hid in cycle_ids]

    def get_vertex_faces(self, vertex_id: VertexId) -> List[Face]:
        """Gets all faces adjacent to a given vertex."""
        v = self.get_proxy(vertex_id)
        if not v:
            return []

        faces = {he.face for he in v.half_edges if he.face}
        return list(faces)

    # --- Property Lookup API ---

    @lru_cache(maxsize=None)
    def get_parents(self, element_id: ElementId) -> Tuple[ElementId, ...]:
        """
        Gets the parent IDs for an element. For surgical commits, this is an
        explicit property. For annotation commits, it is implicit.
        """
        element_type = self._get_element_type_string(element_id)

        explicit_parents = self._prop_manager.get_property(
            element_type, element_id, "parent_ids"
        )
        if explicit_parents is not None:
            return explicit_parents

        is_annotation = (
            len(self._commit.parent_ids) == 1
            and self._factory._commits[self._commit.parent_ids[0]].topology_id
            == self._commit.topology_id
        )

        if is_annotation:
            parent_surface_id = self._commit.parent_ids[0]
            implicit_parent_id = element_id.__class__(
                parent_surface_id, element_id._local_id
            )
            return (implicit_parent_id,)

        return tuple()

    def _get_element_type_string(self, element_id: ElementId) -> str:
        """Converts a CamelCase class name like HalfEdgeId to snake_case like half_edge."""
        cls = element_id.__class__
        if cls in _ELEMENT_TYPE_STRING_CACHE:
            return _ELEMENT_TYPE_STRING_CACHE[cls]

        class_name = cls.__name__.replace("Id", "")
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        _ELEMENT_TYPE_STRING_CACHE[cls] = snake_case
        return snake_case

    def _find_property_iterative(
        self, start_surface_id: SurfaceId, start_element_id: ElementId, prop_name: str
    ) -> Any:
        """
        Iteratively searches for a property by traversing the element's ancestry,
        avoiding recursion to prevent stack overflow on deep histories.
        Returns the found value or None if not found.
        Results are cached at the factory level.
        """
        cache_key = (start_surface_id, start_element_id, prop_name)
        if cache_key in self._factory._property_find_cache:
            return self._factory._property_find_cache[cache_key]

        stack = [(start_surface_id, start_element_id)]
        visited = set()

        while stack:
            current_sid, current_eid = stack.pop()

            state = (current_sid, current_eid)
            if state in visited:
                continue
            visited.add(state)

            # Create a viewer for the current state to access its context.
            current_viewer = self._factory.create_viewer(current_sid)

            if current_eid in current_viewer._commit.tombstones:
                continue

            # 1. Check for a locally defined property value.
            element_type = current_viewer._get_element_type_string(current_eid)
            local_value = current_viewer._prop_manager.get_property(
                element_type, current_eid, prop_name
            )
            if local_value is not None:
                self._factory._property_find_cache[cache_key] = local_value
                return local_value

            # 2. If not found, get the parents of the element and add them to the stack.
            parent_eids = current_viewer.get_parents(current_eid)

            # Add parents to the stack in reverse order to maintain depth-first search.
            for parent_eid in reversed(parent_eids):
                parent_state = (parent_eid.surface_id, parent_eid)
                if parent_state not in visited:
                    stack.append(parent_state)

        # If the loop completes, the property was not found in the ancestry.
        self._factory._property_find_cache[cache_key] = None
        return None

    @lru_cache(maxsize=None)
    def get_property(
        self, element_id: ElementId, prop_name: str, default: Any = None
    ) -> Any:
        """
        Gets a property for this element, searching through history. This result
        is cached.
        """
        value = self._find_property_iterative(self._surface_id, element_id, prop_name)

        return value if value is not None else default
