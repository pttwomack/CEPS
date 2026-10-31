# mazedemo/glue.py
"""
Implements the `glue` operation, the topological inverse of `cut`.

This module provides the functionality to take one or more surfaces with
unsealed boundaries and stitch them together into a new, single manifold.
"""

from typing import List, Tuple, Dict, Set
from absl import logging
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId, HalfEdgeId, VertexId, ElementId
from mazedemo.viewer import Viewer, HalfEdge
from mazedemo.surface import validate_topology
from dataclasses import dataclass


@dataclass(frozen=True)
class _Corner:
    """A dataclass to hold information about a corner of an interface cycle."""

    apex_v_id: VertexId
    cycle_he_id: HalfEdgeId
    fan_he_ids: Tuple[HalfEdgeId, ...]


@dataclass(frozen=True)
class _GluePlan:
    """A declarative plan for a single glue/stitch operation."""

    seam1: List[_Corner]
    seam2: List[_Corner]
    footprint: Dict[str, Set[ElementId]]
    trace_ids: Set[ElementId]


def _analyze_interface(view: Viewer, seed_id: HalfEdgeId) -> List[_Corner]:
    """Analyzes the interface using the read-only Viewer API."""
    interface_hes = {he for he in view.half_edges if he.twin is None}

    start_he = view.get_proxy(seed_id)
    he_cycle, curr = [start_he], start_he
    while len(he_cycle) < len(interface_hes) + 1:
        next_he = curr.next
        while next_he.twin:
            next_he = next_he.twin.next
        if next_he == start_he:
            break
        he_cycle.append(next_he)
        curr = next_he
    else:
        assert all(he.twin is None for he in he_cycle)
        raise RuntimeError(f"cycle not found: {he_cycle}")
    logging.vlog(2, "Found boundary he cycle of length %d: %s", len(he_cycle), he_cycle)

    corner_cycle = []
    for i, he_out in enumerate(he_cycle):
        he_in = he_cycle[i - 1]

        apex_v_id = he_out.origin.id

        fan_hes = []
        curr_fan = he_in.next
        while curr_fan != he_out:
            fan_hes.append(curr_fan)
            curr_fan = curr_fan.twin.next

        assert all(he_out.origin == fan_he.origin for fan_he in fan_hes)
        corner_cycle.append(
            _Corner(
                apex_v_id=apex_v_id,
                cycle_he_id=he_out.id,
                fan_he_ids=tuple([he.id for he in fan_hes]),
            )
        )
    return corner_cycle


def _execute_glue_plan(workbench, plan: _GluePlan):
    """Executes a pre-computed glue plan on a surgery workbench."""
    seam1, seam2 = plan.seam1, plan.seam2[::-1]
    cycle_length = len(seam1)
    assert len(seam1) == len(seam2)

    # Get workbench elements from the plan's IDs
    wb_seam1_hes = [workbench.get_descendant(c.cycle_he_id) for c in seam1]
    wb_seam2_hes = [workbench.get_descendant(c.cycle_he_id) for c in seam2]
    wb_seam1_vs = [workbench.get_descendant(c.apex_v_id) for c in seam1]
    wb_seam2_vs = [workbench.get_descendant(c.apex_v_id) for c in seam2]
    logging.vlog(9, "wb_seam1_vs: %s", wb_seam1_vs)
    logging.vlog(9, "wb_seam2_vs: %s", wb_seam2_vs)

    # Create new, merged vertices
    new_verts = [
        workbench.new_vertex(
            parent_ids=tuple(
                set(wb_seam1_vs[i].parent_ids)
                | set(wb_seam2_vs[(i - 1) % cycle_length].parent_ids)
            )
        )
        for i in range(cycle_length)
    ]
    logging.vlog(9, "created %s", new_verts)

    # Stitch the seams
    for i in range(len(seam1)):
        new_v_1 = new_verts[i]
        new_v_2 = new_verts[(i + 1) % cycle_length]
        he1 = wb_seam1_hes[i]
        he2 = wb_seam2_hes[i]

        # Re-parent the cycle edges from both sides
        assert set(he1.origin.parent_ids) & set(new_v_1.parent_ids)
        he1.origin = new_v_1
        assert set(he2.origin.parent_ids) & set(new_v_2.parent_ids)
        he2.origin = new_v_2
        new_v_1.half_edge = he1

        # Re-parent all fan edges from both sides
        for fan_he_id in seam1[i].fan_he_ids:
            wb_fan_he = workbench.get_descendant(fan_he_id)
            assert set(wb_fan_he.origin.parent_ids) & set(new_v_1.parent_ids)
            wb_fan_he.origin = new_v_1
        for fan_he_id in seam2[i].fan_he_ids:
            wb_fan_he = workbench.get_descendant(fan_he_id)
            assert set(wb_fan_he.origin.parent_ids) & set(new_v_2.parent_ids)
            wb_fan_he.origin = new_v_2

        # Twin the edges to zip the seam
        he1.twin = he2
        he2.twin = he1

    # Delete the old, unmerged vertices
    logging.vlog(9, "deleting %s", wb_seam1_vs + wb_seam2_vs)
    for v in set(wb_seam1_vs + wb_seam2_vs):
        workbench.delete_vertex(v)


def _stitch_one_cycle(
    factory: SurfaceFactory,
    surface_id: SurfaceId,
    current_seed1: HalfEdgeId,
    current_seed2: HalfEdgeId,
) -> SurfaceId:
    """Performs the atomic surgery to glue a single pair of boundary cycles."""
    view = factory.create_viewer(surface_id)
    seam1 = _analyze_interface(view, current_seed1)
    seam2 = _analyze_interface(view, current_seed2)

    if set(c.cycle_he_id for c in seam1) & set(c.cycle_he_id for c in seam2):
        raise ValueError("Cannot glue a boundary cycle to itself.")

    if len(seam1) != len(seam2):
        raise ValueError(
            f"Cannot stitch cycles of unequal length ({len(seam1)} vs {len(seam2)})."
        )

    all_corners = seam1 + seam2
    footprint_hes = {c.cycle_he_id for c in all_corners} | {
        fid for c in all_corners for fid in c.fan_he_ids
    }
    footprint_vs = {c.apex_v_id for c in all_corners}
    footprint = {"half_edge": footprint_hes, "vertex": footprint_vs}
    trace_ids = footprint["half_edge"] | footprint["vertex"]

    plan = _GluePlan(seam1=seam1, seam2=seam2, footprint=footprint, trace_ids=trace_ids)

    with factory.update_topology([surface_id], trace_ids=plan.trace_ids) as surgery:
        _execute_glue_plan(surgery.workbench, plan)
        assert surgery.workbench.is_valid(allow_missing_twin=True)
        logging.vlog(9, "committing %s", surgery.workbench)
        return surgery.commit(f"Glued seam from {current_seed1} to {current_seed2}")


def glue(
    factory: SurfaceFactory,
    surface_ids: List[SurfaceId],
    boundary_seed_pairing: List[Tuple[HalfEdgeId, HalfEdgeId]],
) -> SurfaceId:
    """
    Performs a topological glue operation on one or more oriented surfaces.
    """
    if not surface_ids:
        raise ValueError("Must provide at least one surface ID to glue.")

    with factory.update_topology(surface_ids) as surgery:
        surface_id = surgery.commit("Unify surfaces for gluing.")
        logging.vlog(2, "Unified %s --> %s", surface_ids, surface_id)

    for pair in boundary_seed_pairing:
        d = factory.trace_elements([surface_id], set(pair))
        s1, s2 = next(iter(d[pair[0]])), next(iter(d[pair[1]]))
        new_surface_id = _stitch_one_cycle(factory, surface_id, s1, s2)
        logging.vlog(2, "Stitched %s -> %s", surface_id, new_surface_id)
        surface_id = new_surface_id

    final_viewer = factory.create_viewer(surface_id)
    if not validate_topology(final_viewer, allow_missing_twin=True):
        raise RuntimeError(
            f"glue() operation produced a topologically invalid or unsealed surface: {surface_id}"
        )

    logging.vlog(
        10, "--- State of Surface Immediately After glue() ---\n%s", final_viewer
    )

    return surface_id
