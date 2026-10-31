# mazedemo/surface_factory.py
"""
Implements the SurfaceFactory, the central repository for creating and
managing versioned, immutable surfaces and their properties.
"""

import uuid
import collections
from typing import Any, Dict, Hashable, List, Set, Tuple, Type
from contextlib import contextmanager
import networkx as nx
from absl import logging

from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId
from mazedemo.surface import FrozenSurface, Surface
from mazedemo.viewer import Viewer
from mazedemo.surgery import Surgery
from mazedemo.annotation import Annotation
from mazedemo.surface_property_manager import SurfacePropertyManager
from mazedemo.surface_util import has_parallel_edges

# A commit is a record of a surface's state. It points to a topology
# and a set of parent commits.
Commit = collections.namedtuple("Commit", ["parent_ids", "topology_id", "tombstones"])


class SurfaceFactory:
    """
    The central repository for creating and managing versioned surfaces.
    It provides transactional contexts (Surgery, Annotation) for all mutations.
    """

    def __init__(self):
        # Maps a SurfaceId (commit) to its parent(s) and topology
        self._commits: Dict[SurfaceId, Commit] = {}
        # Maps a topology ID (hash or UUID) to the immutable FrozenSurface data
        self._topologies: Dict[Hashable, FrozenSurface] = {}
        # Manages all property data, keyed by SurfaceId and ElementId
        self._prop_manager = SurfacePropertyManager()
        # Caches Viewer objects to avoid redundant object creation
        self._viewer_cache: Dict[SurfaceId, Viewer] = {}
        # Caches dynamically generated proxy classes for Viewers with custom properties.
        self._proxy_class_cache: Dict[tuple, Dict[str, type]] = {}
        # Cache for the commit graph to avoid rebuilding it repeatedly
        self._commit_graph: nx.DiGraph = None
        # Cache for the expensive _find_property_iterative function in Viewer.
        self._property_find_cache: Dict[tuple, Any] = {}

    def _build_commit_graph(self):
        """
        Builds a networkx DiGraph of the commit history.
        Edges point from parent to child.
        """
        if self._commit_graph is None:
            self._commit_graph = nx.DiGraph()
            for commit_id, commit_data in self._commits.items():
                self._commit_graph.add_node(commit_id)
                for parent_id in commit_data.parent_ids:
                    self._commit_graph.add_edge(parent_id, commit_id)
        return self._commit_graph

    def _get_sorted_relevant_commits(
        self, source_surface_ids: Set[SurfaceId], target_surface_ids: List[SurfaceId]
    ) -> List[SurfaceId]:
        """
        Finds all commits on any path between the source and target surfaces
        and returns them in topological order (parents before children).
        """
        g = self._build_commit_graph()

        all_descendants_of_sources = set(source_surface_ids)
        for source_id in source_surface_ids:
            if source_id in g:
                all_descendants_of_sources.update(nx.descendants(g, source_id))

        all_ancestors_of_targets = set(target_surface_ids)
        for target_id in target_surface_ids:
            if target_id in g:
                all_ancestors_of_targets.update(nx.ancestors(g, target_id))

        relevant_commits = all_descendants_of_sources.intersection(
            all_ancestors_of_targets
        )
        if not relevant_commits:
            return []

        subgraph = g.subgraph(relevant_commits)

        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            return []

    def trace_elements(
        self, target_surface_ids: List[SurfaceId], ancestor_element_ids: Set[ElementId]
    ) -> Dict[ElementId, Set[ElementId]]:
        """
        Traces a set of ancestor elements forward through the commit DAG to
        find all their corresponding descendants on a set of target surfaces.
        """
        if not ancestor_element_ids:
            return {}
        if not target_surface_ids:
            return {eid: set() for eid in ancestor_element_ids}

        source_surface_ids = {eid.surface_id for eid in ancestor_element_ids}
        sorted_commits = self._get_sorted_relevant_commits(
            source_surface_ids, target_surface_ids
        )
        logging.vlog(7, "sorted commits: %s", sorted_commits)

        descendant_map = {eid: {eid} for eid in ancestor_element_ids}

        for commit_id in sorted_commits:
            if commit_id in source_surface_ids:
                continue

            current_commit = self._commits[commit_id]
            parent_ids = current_commit.parent_ids

            is_annotation_commit = (
                len(parent_ids) == 1
                and self._commits[parent_ids[0]].topology_id
                == current_commit.topology_id
            )
            logging.vlog(
                7,
                "Processing %s commit %s",
                "annotation" if is_annotation_commit else "surgery",
                commit_id,
            )

            if is_annotation_commit:
                parent_id = parent_ids[0]
                for source_eid, descendant_eids in descendant_map.items():
                    to_add = set()
                    for desc in descendant_eids:
                        if desc.surface_id == parent_id:
                            to_add.add(desc.__class__(commit_id, desc._local_id))
                    descendant_eids.update(to_add)
            else:  # Surgical Commit
                parent_to_child_map = collections.defaultdict(list)
                viewer = self.create_viewer(commit_id)
                for elements_attr in ["vertices", "half_edges", "faces"]:
                    for element in getattr(viewer, elements_attr):
                        for parent in element.parent_ids:
                            parent_to_child_map[parent].append(element.id)
                logging.vlog(
                    8,
                    "parent_to_child_map for %s (parents=%s): %s",
                    commit_id,
                    self._commits[commit_id].parent_ids,
                    parent_to_child_map,
                )

                for source_eid, descendant_eids in descendant_map.items():
                    children_of_descendants = set()
                    parents_in_this_commit = {
                        p for p in descendant_eids if p.surface_id in parent_ids
                    }
                    logging.vlog(
                        9, "parents in this commit: %s", parents_in_this_commit
                    )

                    for d_eid in parents_in_this_commit:
                        logging.vlog(10, "update children")
                        children_of_descendants.update(
                            parent_to_child_map.get(d_eid, [])
                        )

                    logging.vlog(9, "new children: %s", children_of_descendants)
                    descendant_eids.update(children_of_descendants)

            if current_commit.tombstones:
                for source_eid, descendant_eids in descendant_map.items():
                    descendant_eids.difference_update(current_commit.tombstones)
            logging.vlog(8, "descendant_map after %s: %s", commit_id, descendant_map)

        final_descendants = {eid: set() for eid in ancestor_element_ids}
        for ancestor, descendants in descendant_map.items():
            final_descendants[ancestor] = {
                d for d in descendants if d.surface_id in target_surface_ids
            }

        logging.vlog(4, "Final trace results: %s", final_descendants)
        return final_descendants

    @contextmanager
    def update_topology(
        self,
        source_ids: List[SurfaceId] = None,
        footprint: Dict = None,
        trace_ids: Set[ElementId] = None,
    ) -> Surgery:
        """
        Provides a transactional context (Surgery) for performing topological
        mutations.
        """
        source_ids = source_ids or []
        yield Surgery(self, source_ids, footprint, trace_ids)

    @contextmanager
    def update_properties(self, source_id: SurfaceId) -> Annotation:
        """
        Provides a transactional context (Annotation) for adding or modifying
        properties.
        """
        yield Annotation(self, source_id)

    def create_viewer(
        self, surface_id: SurfaceId, custom_properties: Dict[str, Any] = None
    ) -> Viewer:
        """
        Creates a read-only Viewer for a specific surface version.
        Viewers are cached for efficiency.
        """
        if surface_id not in self._commits:
            raise ValueError(
                f"Cannot create viewer for non-existent surface ID: {surface_id}"
            )

        custom_properties = custom_properties or {}
        cache_key = (surface_id, tuple(sorted(custom_properties.items())))

        if cache_key not in self._viewer_cache:
            # The factory now passes the custom properties to the Viewer's constructor.
            self._viewer_cache[cache_key] = Viewer(self, surface_id, custom_properties)

        return self._viewer_cache[cache_key]

    def get_topology_id(self, surface_id: SurfaceId) -> Hashable:
        """Returns the opaque, unique ID of the topology for a given surface commit."""
        if surface_id not in self._commits:
            raise ValueError(f"Surface ID {surface_id} not found in factory.")
        return self._commits[surface_id].topology_id

    def _commit_surgery_changes(
        self, surgery: Surgery, final_workbench: Surface, commit_message: str
    ) -> SurfaceId:
        """
        Internal commit logic for a Surgery transaction. Creates a new
        topology and a new commit, correctly handling provenance for all elements.
        """
        # Check for the introduction of parallel edges.
        source_had_parallel = any(
            has_parallel_edges(self.create_viewer(sid)) for sid in surgery._source_ids
        )
        final_has_parallel = has_parallel_edges(final_workbench)

        if final_has_parallel and not source_had_parallel:
            logging.warning(
                "Surgery '%s' introduced parallel edges to a surface that did not previously have them. "
                "This may indicate an unintended topological change.",
                commit_message,
            )

        self._commit_graph = None  # Invalidate graph cache
        frozen_surface = final_workbench.to_frozen()
        topology_id = uuid.uuid4()
        self._topologies[topology_id] = frozen_surface

        new_surface_id = SurfaceId(uuid.uuid4())

        tombstones = set()
        if surgery._is_partition:
            tombstones = self._get_tombstoned_ids(surgery)

        self._commits[new_surface_id] = Commit(
            parent_ids=tuple(surgery._source_ids),
            topology_id=topology_id,
            tombstones=frozenset(tombstones),
        )

        # Explicitly set parent_ids for elements that were part of the surgery.
        # FIX: We MUST filter out deleted elements before enumerating.
        # The 'to_frozen' method creates a compact array excluding deleted items.
        # The Viewer uses indices into that compact array.
        # Therefore, the ElementIds we assign properties to here must correspond
        # to that same compact sequence.

        retained_parent_ids = set()

        live_vertices = [
            v
            for v in final_workbench.vertices
            if v is not None and not getattr(v, "is_deleted", False)
        ]
        for i, v in enumerate(live_vertices):
            v_id = VertexId(new_surface_id, i)
            self._prop_manager.set_property(
                "vertex", v_id, "parent_ids", tuple(v.parent_ids)
            )
            retained_parent_ids.update(v.parent_ids)

        live_faces = [
            f
            for f in final_workbench.faces
            if f is not None and not getattr(f, "is_deleted", False)
        ]
        for i, f in enumerate(live_faces):
            f_id = FaceId(new_surface_id, i)
            self._prop_manager.set_property(
                "face", f_id, "parent_ids", tuple(f.parent_ids)
            )
            retained_parent_ids.update(f.parent_ids)

        live_half_edges = [
            he
            for he in final_workbench.half_edges
            if he is not None and not getattr(he, "is_deleted", False)
        ]
        for i, he in enumerate(live_half_edges):
            he_id = HalfEdgeId(new_surface_id, i)
            self._prop_manager.set_property(
                "half_edge", he_id, "parent_ids", tuple(he.parent_ids)
            )
            retained_parent_ids.update(he.parent_ids)

        # Handle "pass-through" elements for single-parent, non-partitioning surgeries with a footprint.
        if (
            len(surgery._source_ids) == 1
            and surgery._footprint
            and not surgery._is_partition
        ):
            source_id = list(surgery._source_ids)[0]
            source_viewer = self.create_viewer(source_id)
            for el_type, id_list_attr in [
                ("vertex", "vertex_ids"),
                ("face", "face_ids"),
                ("half_edge", "half_edge_ids"),
            ]:
                for element_id in getattr(source_viewer, id_list_attr):
                    if element_id not in retained_parent_ids:
                        new_element_id = element_id.__class__(
                            new_surface_id, element_id._local_id
                        )
                        self._prop_manager.set_property(
                            el_type, new_element_id, "parent_ids", (element_id,)
                        )

        return new_surface_id

    def _get_tombstoned_ids(self, surgery: Surgery) -> Set[Hashable]:
        """
        For a partition operation, finds all elements in the source surfaces
        that were *not* carried over.
        """
        retained_parent_ids = set()
        for elem_list in [
            surgery.workbench.vertices,
            surgery.workbench.faces,
            surgery.workbench.half_edges,
        ]:
            for elem in elem_list:
                retained_parent_ids.update(elem.parent_ids)

        tombstoned_ids = set()
        for source_id in surgery._source_ids:
            source_viewer = self.create_viewer(source_id)
            for id_list in [
                source_viewer.vertex_ids,
                source_viewer.face_ids,
                source_viewer.half_edge_ids,
            ]:
                for element_id in id_list:
                    if element_id not in retained_parent_ids:
                        tombstoned_ids.add(element_id)
        return tombstoned_ids

    def _commit_annotation_changes(
        self, source_id: SurfaceId, staged_properties: Dict, commit_message: str
    ) -> SurfaceId:
        """
        Internal commit logic for an Annotation transaction. Creates a new
        commit that points to an existing topology.
        """
        self._commit_graph = None  # Invalidate graph cache
        parent_commit = self._commits[source_id]
        parent_viewer = self.create_viewer(source_id)

        new_surface_id = SurfaceId(uuid.uuid4())
        # Annotation commits have no tombstones.
        self._commits[new_surface_id] = Commit(
            parent_ids=(source_id,),
            topology_id=parent_commit.topology_id,
            tombstones=frozenset(),
        )

        # Store the new property values in the property manager
        for prop_name, element_types in staged_properties.items():
            for element_type, elements in element_types.items():
                for element_id, value in elements.items():
                    new_element_id = element_id.__class__(
                        new_surface_id, element_id._local_id
                    )
                    self._prop_manager.set_property(
                        element_type, new_element_id, prop_name, value
                    )

        return new_surface_id
