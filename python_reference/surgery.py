# mazedemo/surgery.py
"""
Defines the Surgery class, a transactional context for creating a single new
surface from one or more source surfaces by altering topology.
"""

import re
from typing import Any, Dict, Hashable, List, Set, Tuple, Type
import numpy as np
from absl import logging
import collections

from mazedemo.surface import Surface, FrozenSurface, Vertex, Face, HalfEdge
from mazedemo.ids import SurfaceId, VertexId, FaceId, HalfEdgeId, ElementId


# Forward declaration to avoid circular import
class SurfaceFactory:
    pass


class Surgery:
    """
    A transactional context for creating a single new surface from one or more
    source surfaces. It provides a single, unified workbench for surgery.
    """

    def __init__(
        self,
        factory: "SurfaceFactory",
        source_ids: List[SurfaceId],
        footprint: Dict = None,
        trace_ids: Set[ElementId] = None,
    ):
        self._factory = factory
        self._source_ids = frozenset(source_ids)
        self._footprint = footprint
        self._trace_ids = trace_ids
        self._is_partition = False

        # +++ START DEBUGGING LOG +++
        logging.vlog(
            8,
            f"--- Surgery Init ---\n  Sources: {self._source_ids}\n  trace_ids provided? {self._trace_ids is not None}",
        )
        if self._trace_ids:
            logging.vlog(8, f"  Number of trace_ids: {len(self._trace_ids)}")
        # +++ END DEBUGGING LOG +++

        self._workbench: Surface = self._create_unified_workbench()
        self._workbench.source_surface_ids = self._source_ids

        if self._trace_ids:
            self._trace_and_populate_workbench()

    def _trace_and_populate_workbench(self):
        """
        Traces specified ancestors to their descendants in the source surfaces,
        then finds their final descendants in the newly created workbench and
        populates the workbench's trace results.
        """
        if not self._source_ids:
            self._workbench.traced_descendants = {}
            return

        # 1. Trace ancestors to the immediate source surfaces of this surgery.
        descendants_on_sources = self._factory.trace_elements(
            target_surface_ids=list(self._source_ids),
            ancestor_element_ids=self._trace_ids,
        )

        # 2. Map from the original ancestors to their final descendants in the workbench.
        final_descendants = collections.defaultdict(set)
        logging.vlog(9, "_parent_id_map: %s", self._workbench._parent_id_map)
        for ancestor_id, source_descendants in descendants_on_sources.items():
            for source_descendant_id in source_descendants:
                # Use the workbench's internal parent map to find the final descendants
                workbench_descendants = self._workbench._parent_id_map.get(
                    source_descendant_id, []
                )
                final_descendants[ancestor_id].update(workbench_descendants)

        # 3. Store the results on the workbench itself.
        logging.vlog(10, "Populated traced_descendants: %s", final_descendants)
        self._workbench.traced_descendants = dict(final_descendants)

    def set_as_partition(self):
        """
        Indicates that the final commit should only contain elements that were
        part of the original footprint, plus any newly created elements.
        Untouched elements from the source surface(s) that were outside the
        footprint will be discarded (and implicitly tombstoned).
        """
        if not self._footprint:
            raise ValueError(
                "set_as_partition() can only be called when a footprint is provided."
            )
        self._is_partition = True
        # Rebuild the workbench now that the partition flag is set.
        self._workbench = self._create_unified_workbench()

    def _resolve_footprint(self) -> Dict:
        """
        Resolves a footprint that may contain ancestor IDs by tracing them to the
        immediate source surfaces.
        """
        if not self._footprint:
            return None

        resolved_footprint = collections.defaultdict(set)
        ancestor_ids_to_trace = collections.defaultdict(set)

        # Separate direct source IDs from ancestor IDs that need tracing
        for el_type, id_set in self._footprint.items():
            for element_id in id_set:
                if element_id.surface_id in self._source_ids:
                    resolved_footprint[el_type].add(element_id)
                else:
                    ancestor_ids_to_trace[el_type].add(element_id)

        # If there are any ancestor IDs, trace them
        all_ancestors = {
            id for id_set in ancestor_ids_to_trace.values() for id in id_set
        }
        if all_ancestors:
            logging.vlog(
                10, "Resolving footprint, tracing ancestors: %s", all_ancestors
            )
            trace_results = self._factory.trace_elements(
                target_surface_ids=list(self._source_ids),
                ancestor_element_ids=all_ancestors,
            )
            logging.vlog(10, "Trace results for footprint: %s", trace_results)
            # Add the traced descendants to the resolved footprint
            for ancestor_id, descendants in trace_results.items():
                if len(descendants) != 1:
                    raise RuntimeError(
                        f"Footprint trace found {len(descendants)} descendants for {ancestor_id}, expected 1."
                    )
                class_name = ancestor_id.__class__.__name__.replace("Id", "")
                el_type_str = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
                resolved_footprint[el_type_str].update(descendants)

        return dict(resolved_footprint)

    def _create_unified_workbench(self) -> Surface:
        """
        Merges and reconstructs the object graph for all source surfaces into
        a single, unified workbench. If a footprint is provided and this is a
        partition, it constructs a partial workbench.
        """
        workbench = Surface()
        if not self._source_ids:
            return workbench

        resolved_footprint = self._resolve_footprint()
        logging.vlog(
            10, "Creating workbench with resolved_footprint: %s", resolved_footprint
        )

        for source_id in self._source_ids:
            commit = self._factory._commits[source_id]
            frozen = self._factory._topologies[commit.topology_id]

            if resolved_footprint and self._is_partition:
                self._build_partial_component(
                    workbench, source_id, frozen, resolved_footprint
                )
            else:
                self._build_full_component(workbench, source_id, frozen)

        workbench.__post_init__()
        logging.vlog(
            10, "Workbench parent_id_map keys: %s", workbench._parent_id_map.keys()
        )
        return workbench

    def _build_partial_component(
        self, workbench: Surface, sid: SurfaceId, frozen: FrozenSurface, footprint: Dict
    ):
        """
        Constructs a partial Surface workbench containing only the elements
        specified in the footprint, ensuring the edit is safely scoped.
        """
        # Get the local indices of elements to keep from the footprint ElementIds
        v_indices_to_keep = {
            v_id._local_id
            for v_id in footprint.get("vertex", set())
            if v_id.surface_id == sid
        }
        f_indices_to_keep = {
            f_id._local_id
            for f_id in footprint.get("face", set())
            if f_id.surface_id == sid
        }
        he_indices_to_keep = {
            he_id._local_id
            for he_id in footprint.get("half_edge", set())
            if he_id.surface_id == sid
        }

        # Maps for old local index to new workbench object
        vert_map, face_map, he_map = {}, {}, {}

        # --- First Pass: Create only the specified objects ---
        for v_idx in v_indices_to_keep:
            vert_map[v_idx] = workbench.new_vertex(parent_ids=[VertexId(sid, v_idx)])

        for f_idx in f_indices_to_keep:
            face_map[f_idx] = workbench.new_face(parent_ids=[FaceId(sid, f_idx)])

        for he_idx in he_indices_to_keep:
            origin_idx = frozen.half_edges[he_idx, 0]
            if origin_idx in vert_map:
                he = workbench.new_half_edge(
                    origin=vert_map[origin_idx], parent_ids=[HalfEdgeId(sid, he_idx)]
                )
                he_map[he_idx] = he

        # --- Second Pass: Wire up pointers for the kept elements ---
        for he_idx, he in he_map.items():
            he_data = frozen.half_edges[he_idx]
            twin_idx, next_idx, prev_idx, face_idx = (
                he_data[1],
                he_data[2],
                he_data[3],
                he_data[4],
            )

            # Only link to elements that are also in the footprint
            if twin_idx in he_map:
                he.twin = he_map[twin_idx]
            if next_idx in he_map:
                he.next = he_map[next_idx]
            if prev_idx in he_map:
                he.prev = he_map[prev_idx]
            if face_idx in face_map:
                he.face = face_map[face_idx]

        # --- Third Pass: Set entry points ---
        for v_idx, v in vert_map.items():
            he_idx = frozen.vertices[v_idx, 0]
            if he_idx in he_map:
                v.half_edge = he_map[he_idx]

        for f_idx, f in face_map.items():
            he_idx = frozen.faces[f_idx, 0]
            if he_idx in he_map:
                f.half_edge = he_map[he_idx]

        all_new_hes = list(he_map.values())
        for v_obj in vert_map.values():
            if v_obj.half_edge is None:
                for he_obj in all_new_hes:
                    if he_obj.origin == v_obj:
                        v_obj.half_edge = he_obj
                        break

        for f_obj in face_map.values():
            if f_obj.half_edge is None:
                for he_obj in all_new_hes:
                    if he_obj.face == f_obj:
                        f_obj.half_edge = he_obj
                        break

        logging.vlog(
            2,
            "_build_partial_component: workbench has V=%d, E=%d, F=%d",
            len(workbench.vertices),
            len(workbench.half_edges),
            len(workbench.faces),
        )
        if (
            len(workbench.vertices) == 0
            and len(workbench.half_edges) == 0
            and len(workbench.faces) == 0
            and any(footprint.values())
        ):
            raise RuntimeError("empty workbench")

    def _build_full_component(
        self, workbench: Surface, source_id: SurfaceId, frozen: FrozenSurface
    ):
        """Helper to build a full component from a FrozenSurface."""
        num_verts = frozen.vertices.shape[0]
        num_faces = frozen.faces.shape[0]
        num_hes = frozen.half_edges.shape[0]

        source_verts = [
            workbench.new_vertex(parent_ids=(VertexId(source_id, i),))
            for i in range(num_verts)
        ]
        source_faces = [
            workbench.new_face(parent_ids=(FaceId(source_id, i),))
            for i in range(num_faces)
        ]

        temp_hes = []
        for i in range(num_hes):
            origin_v = source_verts[frozen.half_edges[i, 0]]
            he = workbench.new_half_edge(
                origin=origin_v, parent_ids=(HalfEdgeId(source_id, i),)
            )
            temp_hes.append(he)

        for i, he_data in enumerate(frozen.half_edges):
            he = temp_hes[i]
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
                he.face = source_faces[face_idx]

        for i, v_data in enumerate(frozen.vertices):
            he_idx = v_data[0]
            if he_idx != -1:
                source_verts[i].half_edge = temp_hes[he_idx]

        for i, f_data in enumerate(frozen.faces):
            he_idx = f_data[0]
            if he_idx != -1:
                source_faces[i].half_edge = temp_hes[he_idx]
        logging.vlog(
            2,
            "_build_full_component: workbench has V=%d, E=%d, F=%d",
            len(workbench.vertices),
            len(workbench.half_edges),
            len(workbench.faces),
        )

    @property
    def workbench(self) -> Surface:
        """
        Gets the mutable, unified workbench for this surgical operation.
        """
        return self._workbench

    def commit(self, commit_message: str) -> SurfaceId:
        """
        Finalizes the transaction by converting the workbench to a FrozenSurface
        and committing it to the factory.
        """
        return self._factory._commit_surgery_changes(
            self, self._workbench, commit_message
        )
