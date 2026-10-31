# mazedemo/annotation.py
"""
Defines the Annotation class and its associated writable proxy views for
staging property changes within a transaction.
"""

from typing import Any, Dict, Hashable, List, Set, Tuple, Type, ContextManager, Optional
from contextlib import contextmanager
import collections

from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId
from mazedemo.viewer import Viewer
from mazedemo.surface_util import get_topology_str


# Forward declarations to avoid circular imports
class SurfaceFactory:
    pass


# --- Writable Proxy Views (Internal Implementation) ---


class _WritableElement:
    """A base proxy object for a surface element that allows property setting."""

    _type: str  # To be defined by subclasses

    def __init__(self, annotation: "Annotation", element_id: ElementId):
        self._annotation = annotation
        self.id = element_id

    def __setattr__(self, name: str, value: Any):
        # Allow setting internal attributes and properties.
        if name.startswith("_") or name == "id" or hasattr(self.__class__, name):
            super().__setattr__(name, value)
        else:
            # Intercept all other assignments to stage them as properties.
            self._annotation.stage_property(self._type, self.id, name, value)

    def __getattr__(self, name: str) -> Any:
        """
        First check for staged properties in the current transaction, then
        fall back to the read-only viewer for property access from history.
        """
        # Check staged properties first.
        staged_value = (
            self._annotation._staged_properties.get(name, {})
            .get(self._type, {})
            .get(self.id)
        )
        if staged_value is not None:
            return staged_value

        # Fallback to the viewer for historical properties.
        return self._annotation._viewer.get_property(self.id, name)


class _WritableVertex(_WritableElement):
    """A proxy object for a vertex that allows property setting and traversal."""

    _type = "vertex"

    @property
    def half_edge(self) -> Optional["_WritableHalfEdge"]:
        """An arbitrary half-edge originating from this vertex."""
        return self._annotation._get_writable_half_edge_for_vertex(self.id)

    @property
    def half_edges(self) -> List["_WritableHalfEdge"]:
        """Returns all half-edges originating from this vertex."""
        # Delegate traversal to the underlying viewer
        viewer_vertex = self._annotation._viewer.get_proxy(self.id)
        if not viewer_vertex:
            return []

        # Wrap the resulting IDs in writable proxies.
        return [
            self._annotation._get_writable_proxy(he.id)
            for he in viewer_vertex.half_edges
        ]


class _WritableFace(_WritableElement):
    """A proxy object for a face that allows property setting and traversal."""

    _type = "face"

    @property
    def half_edges(self) -> List["_WritableHalfEdge"]:
        """The cycle of half-edges bounding this face."""
        return self._annotation._get_writable_half_edges_for_face(self.id)

    @property
    def vertices(self) -> List["_WritableVertex"]:
        """Returns all vertices belonging to this face."""
        # Delegate traversal to the underlying viewer
        viewer_face = self._annotation._viewer.get_proxy(self.id)
        if not viewer_face:
            return []

        # Wrap the resulting vertex IDs in writable proxies.
        return [
            self._annotation._get_writable_proxy(v.id) for v in viewer_face.vertices
        ]


class _WritableHalfEdge(_WritableElement):
    """A proxy object for a half-edge that allows property setting and traversal."""

    _type = "half_edge"

    @property
    def origin(self) -> Optional["_WritableVertex"]:
        return self._annotation._get_writable_origin(self.id)

    @property
    def twin(self) -> Optional["_WritableHalfEdge"]:
        return self._annotation._get_writable_twin(self.id)

    @property
    def next(self) -> Optional["_WritableHalfEdge"]:
        return self._annotation._get_writable_next(self.id)

    @property
    def prev(self) -> Optional["_WritableHalfEdge"]:
        return self._annotation._get_writable_prev(self.id)

    @property
    def face(self) -> Optional["_WritableFace"]:
        return self._annotation._get_writable_face(self.id)


class _WritableView:
    """A temporary, writable view yielded by the Annotation.writer()."""

    def __init__(self, annotation: "Annotation"):
        self._annotation = annotation

    def __str__(self):
        return get_topology_str(self)

    def get_proxy(self, element_id: ElementId) -> Optional["_WritableElement"]:
        """Gets or creates a writable proxy object for a given ElementId."""
        return self._annotation._get_writable_proxy(element_id)

    @property
    def vertices(self) -> List[_WritableVertex]:
        """Returns a list of writable proxy objects for each vertex."""
        return [
            self._annotation._get_writable_proxy(vid)
            for vid in self._annotation._viewer.vertex_ids
        ]

    @property
    def faces(self) -> List[_WritableFace]:
        """Returns a list of writable proxy objects for each face."""
        return [
            self._annotation._get_writable_proxy(fid)
            for fid in self._annotation._viewer.face_ids
        ]

    @property
    def half_edges(self) -> List[_WritableHalfEdge]:
        """Returns a list of writable proxy objects for each half-edge."""
        return [
            self._annotation._get_writable_proxy(hid)
            for hid in self._annotation._viewer.half_edge_ids
        ]


# --- The Annotation Class ---


class Annotation:
    """
    A transactional context for adding or modifying properties on an existing,
    unchanged topology.
    """

    def __init__(self, factory: "SurfaceFactory", surface_id: SurfaceId):
        self._factory = factory
        self._surface_id = surface_id
        self._viewer = self._factory.create_viewer(surface_id)
        self._staged_properties: Dict[str, Dict[str, Dict[Hashable, Any]]] = (
            collections.defaultdict(lambda: {"vertex": {}, "face": {}, "half_edge": {}})
        )
        # Internal cache for writable proxies to ensure object identity.
        self._proxy_cache: Dict[ElementId, _WritableElement] = {}

    def _get_writable_proxy(self, element_id: ElementId) -> Optional[_WritableElement]:
        """Gets or creates a writable proxy object for a given ElementId."""
        if element_id is None:
            return None
        if element_id in self._proxy_cache:
            return self._proxy_cache[element_id]

        proxy_class = None
        if isinstance(element_id, VertexId):
            proxy_class = _WritableVertex
        elif isinstance(element_id, FaceId):
            proxy_class = _WritableFace
        elif isinstance(element_id, HalfEdgeId):
            proxy_class = _WritableHalfEdge

        if proxy_class:
            proxy = proxy_class(self, element_id)
            self._proxy_cache[element_id] = proxy
            return proxy
        return None

    # --- Internal Delegation Methods for Writable Proxies ---

    def _get_writable_origin(self, he_id: HalfEdgeId) -> Optional[_WritableVertex]:
        origin_id = (
            self._viewer.get_origin(he_id).id
            if self._viewer.get_origin(he_id)
            else None
        )
        return self._get_writable_proxy(origin_id)

    def _get_writable_twin(self, he_id: HalfEdgeId) -> Optional[_WritableHalfEdge]:
        twin_id = (
            self._viewer.get_twin(he_id).id if self._viewer.get_twin(he_id) else None
        )
        return self._get_writable_proxy(twin_id)

    def _get_writable_next(self, he_id: HalfEdgeId) -> Optional[_WritableHalfEdge]:
        next_id = (
            self._viewer.get_next(he_id).id if self._viewer.get_next(he_id) else None
        )
        return self._get_writable_proxy(next_id)

    def _get_writable_prev(self, he_id: HalfEdgeId) -> Optional[_WritableHalfEdge]:
        prev_id = (
            self._viewer.get_prev(he_id).id if self._viewer.get_prev(he_id) else None
        )
        return self._get_writable_proxy(prev_id)

    def _get_writable_face(self, he_id: HalfEdgeId) -> Optional[_WritableFace]:
        face_id = (
            self._viewer.get_face(he_id).id if self._viewer.get_face(he_id) else None
        )
        return self._get_writable_proxy(face_id)

    def _get_writable_half_edge_for_vertex(
        self, v_id: VertexId
    ) -> Optional[_WritableHalfEdge]:
        he_id = (
            self._viewer.get_vertex_half_edge(v_id).id
            if self._viewer.get_vertex_half_edge(v_id)
            else None
        )
        return self._get_writable_proxy(he_id)

    def _get_writable_half_edges_for_face(
        self, f_id: FaceId
    ) -> List[_WritableHalfEdge]:
        he_ids = [he.id for he in self._viewer.get_face_half_edges(f_id)]
        return [self._get_writable_proxy(hid) for hid in he_ids]

    # --- Core Transactional Methods ---

    def stage_property(
        self, element_type: str, element_id: Hashable, prop_name: str, value: Any
    ):
        """Stages a property change to be written on commit."""
        self._staged_properties[prop_name][element_type][element_id] = value

    @contextmanager
    def writer(self) -> ContextManager[_WritableView]:
        """
        A context manager that yields a writable view, allowing attribute-style
        property setting and topological traversal.
        """
        yield _WritableView(self)

    def commit(self, commit_message: str) -> SurfaceId:
        """
        Finalizes the transaction by passing the staged properties to the
        factory, which creates a new commit.
        """
        return self._factory._commit_annotation_changes(
            source_id=self._surface_id,
            staged_properties=self._staged_properties,
            commit_message=commit_message,
        )
