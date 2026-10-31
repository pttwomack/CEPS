r"""
Implements mesh refinement algorithms based on edge flipping.

This module provides functions to convert a triangle mesh into its
canonical intrinsic Delaunay triangulation using different geometric conditions.

The geometric conditions and algorithms are based on the principles described in:
  Gillespie, M., Springborn, B., & Crane, K. (2021).
  Discrete Conformal Equivalence of Polyhedral Surfaces.
  ACM Transactions on Graphics (TOG), 40(4), 1-20.
"""

import os
import numpy as np
from typing import Callable, Dict
from absl import logging

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.surface_util import (
    get_topology_str,
    validate_topology,
    has_missing_twin,
    has_missing_face,
)
from mazedemo.surface import Surface
from mazedemo.ids import SurfaceId, HalfEdgeId, VertexId
from mazedemo.viewer import HalfEdge, Viewer
from mazedemo.euclidean import (
    calculate_euclidean_flip_score,
    calculate_euclidean_flip_length,
)
from mazedemo.hyperbolic import (
    calculate_hyperbolic_flip_score,
    calculate_hyperbolic_flip_length,
)

# Global debug flag to enable expensive validation checks.
# This should be False for production use.
_DEBUG_VALIDATE_EACH_FLIP = os.getenv("DEBUG_VALIDATE")

# ==============================================================================
# --- Private Helper Functions ---
# ==============================================================================


def _get_canonical_edge_set(view: "Viewer") -> frozenset:
    """Computes a canonical, hashable set of edges for cycle detection."""
    canonical_edges = set()
    for he in view.half_edges:
        if not he.twin:
            continue
        # Use the local_id directly, as we are only concerned with the current topology.
        v1_id = he.origin.id._local_id
        v2_id = he.twin.origin.id._local_id
        if v1_id < v2_id:
            canonical_edges.add(frozenset({v1_id, v2_id}))
    return frozenset(canonical_edges)


def _get_canonical_edge_set_from_workbench(wb: "Surface") -> frozenset:
    """Computes a canonical, hashable set of edges for cycle detection from a workbench."""
    canonical_edges = set()
    for he in wb.half_edges:
        if he.is_deleted or not he.twin or he.twin.is_deleted:
            continue
        v1_id = he.origin.id
        v2_id = he.twin.origin.id
        if v1_id < v2_id:
            canonical_edges.add(frozenset({v1_id, v2_id}))
    return frozenset(canonical_edges)


def can_flip(view, he_id: HalfEdgeId, allow_parallel_edges: bool = False) -> bool:
    """Checks for topological obstacles to flipping an edge."""
    he = view.get_proxy(he_id)
    if not he or not he.twin:
        return False
    v_b, v_d = he.prev.origin, he.twin.prev.origin
    # Don't flip if the opposite vertices are the same (e.g., a triangle fan around a vertex)
    if v_b.id == v_d.id:
        # This configuration is only flippable as a last resort, during the pass
        # that allows creating parallel edges.
        if not allow_parallel_edges:
            return False

    if not allow_parallel_edges:
        # Check if an edge already exists between the opposite vertices.
        if any(h.twin and h.twin.origin.id == v_d.id for h in v_b.half_edges):
            return False

    return True


def _can_flip_on_workbench(he: "HalfEdge", allow_parallel_edges: bool = False) -> bool:
    """Checks for topological obstacles to flipping an edge on a workbench."""
    # An edge can't be flipped if it or its twin has been deleted in this transaction.
    if he.is_deleted or not he.twin or he.twin.is_deleted:
        return False

    # Or if its adjacent faces have been removed.
    if not he.face or he.face.is_deleted or not he.twin.face or he.twin.face.is_deleted:
        return False

    v_b = he.prev.origin
    v_d = he.twin.prev.origin

    # Don't flip if the opposite vertices are the same
    if v_b.id == v_d.id:
        if not allow_parallel_edges:
            return False

    if not allow_parallel_edges:
        # Check if an edge already exists between the opposite vertices.
        # This must check against the current (potentially modified) state of the workbench.
        if any(
            not h.is_deleted
            and h.twin
            and not h.twin.is_deleted
            and h.twin.origin.id == v_d.id
            for h in v_b.half_edges
        ):
            return False
    return True


def _calculate_ideal_flip_score(he: "HalfEdge") -> float:
    r"""
    Calculates the "illegality" score for Ideal Delaunay triangulations (CEPS Eq 10).
    Positive score means the edge is Non-Delaunay.
    Assumes he.length corresponds to the exponentiated Penner coordinate.
    """
    try:
        l_ac, l_ba, l_cb = he.length, he.prev.length, he.next.length
        l_ad, l_cd = he.twin.next.length, he.twin.prev.length
        if any(l is None or l <= 0 for l in [l_ac, l_ba, l_cb, l_ad, l_cd]):
            return -1.0

        # Equation 10 from CEPS:
        # (Note: This condition is theoretically equivalent to the Euclidean cotan
        # condition under the Penner coordinate transform.)
        lhs = (l_ac**2) * (l_cb * l_ba + l_ad * l_cd)
        rhs = (l_ad * l_ba + l_cb * l_cd) * (l_ad * l_cb + l_ba * l_cd)
        return lhs - rhs
    except (AttributeError, TypeError, ValueError):
        return -1.0


def _calculate_ptolemy_flip_length(he: "HalfEdge") -> float:
    """Calculates the new diagonal length using the Ptolemy relation."""
    len_ba, len_ad = he.prev.length, he.twin.next.length
    len_cb, len_cd = he.next.length, he.twin.prev.length
    return (len_ba * len_cd + len_cb * len_ad) / he.length


def _flip_edge_in_workbench(wb: "Surface", he_ac: "HalfEdge"):
    """
    Performs an edge flip directly on a workbench surface.
    This function does NOT handle transaction management or property annotation.
    Returns the two new half-edges that were created.
    """
    he_id_for_logging = he_ac.parent_ids[0]
    allow_twins, allow_faces = False, False
    if _DEBUG_VALIDATE_EACH_FLIP:
        # Dynamically allow boundaries if they already exist on the surface.
        allow_twins = has_missing_twin(wb)
        allow_faces = has_missing_face(wb)
        if not validate_topology(
            wb,
            allow_missing_twin=allow_twins,
            allow_missing_face=allow_faces,
            allow_multi_edge=True,
        ):
            logging.error(
                "Validation failed on surface BEFORE flipping edge %s.",
                he_id_for_logging,
            )
            logging.error(
                "Corrupt surface state before flip:\n%s",
                get_topology_str(wb),
            )
            raise RuntimeError(
                f"Topology validation failed before flipping edge {he_id_for_logging}"
            )
    he_ca = he_ac.twin
    if he_ca is None:
        # This should be pre-validated by the caller.
        raise ValueError(f"Cannot flip a boundary edge: {he_ac.id} has no twin.")

    # The quadrilateral is A-B-C-D, and we are flipping edge AC for BD.
    # he_ac is A->C, he_ca is C->A.
    # Face containing he_ca (C->A) is ABC.
    # Face containing he_ac (A->C) is ADC.

    # 1. Identify all vertices and boundary half-edges.
    v_a = he_ac.origin
    v_c = he_ca.origin

    # he_ca.prev is the edge before C->A in face ABC, which is B->C. Its origin is B.
    he_bc = he_ca.prev
    v_b = he_bc.origin

    # he_ac.prev is the edge before A->C in face ADC, which is D->A. Its origin is D.
    he_da = he_ac.prev
    v_d = he_da.origin

    # The other boundary edges
    he_ab = he_ca.next
    he_cd = he_ac.next

    # Old faces to be deleted
    old_face_abc = he_ca.face
    old_face_adc = he_ac.face

    # 2. Create new elements: edge BD and faces ABD, BCD.
    # The new edge BD inherits the parentage of the old edge AC.
    he_bd = wb.new_half_edge(origin=v_b, parent_ids=he_ac.parent_ids)
    he_db = wb.new_half_edge(origin=v_d, parent_ids=he_ca.parent_ids)
    he_bd.twin, he_db.twin = he_db, he_bd
    new_face_abd = wb.new_face()
    new_face_bcd = wb.new_face()

    # 3. Re-wire pointers for the new face ABD (cycle A->B->D->A).
    new_face_abd.half_edge = he_ab
    he_ab.face, he_bd.face, he_da.face = new_face_abd, new_face_abd, new_face_abd
    he_ab.next, he_bd.next, he_da.next = he_bd, he_da, he_ab
    he_bd.prev, he_da.prev, he_ab.prev = he_ab, he_bd, he_da

    # 4. Re-wire pointers for the new face BCD (cycle B->C->D->B).
    new_face_bcd.half_edge = he_bc
    he_bc.face, he_cd.face, he_db.face = new_face_bcd, new_face_bcd, new_face_bcd
    he_bc.next, he_cd.next, he_db.next = he_cd, he_db, he_bc
    he_cd.prev, he_db.prev, he_bc.prev = he_bc, he_cd, he_db

    # 5. Update the 'half_edge' pointers for the four vertices of the quad.
    v_a.half_edge, v_b.half_edge, v_c.half_edge, v_d.half_edge = (
        he_ab,
        he_bc,
        he_cd,
        he_da,
    )

    # 6. Mark old elements for deletion.
    wb.delete_half_edge(he_ac)
    wb.delete_half_edge(he_ca)
    wb.delete_face(old_face_abc)
    wb.delete_face(old_face_adc)

    if _DEBUG_VALIDATE_EACH_FLIP:
        # The flip operation should not change the boundary status of the mesh.
        # We use the same permissions as the pre-flip check.
        if not validate_topology(
            wb,
            allow_missing_twin=allow_twins,
            allow_missing_face=allow_faces,
            allow_multi_edge=True,
        ):
            logging.error(
                "Validation failed immediately after flipping edge %s.",
                he_id_for_logging,
            )
            logging.error("Corrupt surface state after flip:\n%s", get_topology_str(wb))
            raise RuntimeError(
                f"Topology validation failed after flipping edge {he_id_for_logging}"
            )
    return he_bd, he_db


def flip_edge(
    factory: SurfaceFactory, surface_id: SurfaceId, he_id: HalfEdgeId, new_length: float
) -> SurfaceId:
    """
    Replaces one diagonal of a quadrilateral with the other one.
    Sets the length of the new diagonal.
    """

    with factory.update_topology([surface_id]) as surgery:
        wb = surgery.workbench
        he_ac = wb.get_descendant(he_id)
        if he_ac is None:
            raise ValueError(f"Half-edge {he_id} not found in workbench.")

        _flip_edge_in_workbench(wb, he_ac)

        flipped_id = surgery.commit("Flip edge")

    with factory.update_properties(flipped_id) as annotation:
        with annotation.writer() as new_view:
            # Find the new half-edge by looking for the one that is a descendant
            # of the original half-edge that was flipped.
            new_he = None
            for he in new_view.half_edges:
                if he_id in he.parent_ids:
                    new_he = he
                    break
            if new_he is None:
                raise RuntimeError(
                    f"Could not find descendant for flipped edge {he_id}"
                )
            new_he.length, new_he.twin.length = new_length, new_length
    return annotation.commit("Annotate flipped edge length")


def _refine(
    factory: SurfaceFactory,
    initial_surface_id: SurfaceId,
    score_fn: Callable,
    length_calculation_fn: Callable,
    epsilon: float,
    max_passes: int = None,
) -> SurfaceId:
    """
    Refines the triangulation to satisfy the condition defined by score_fn.
    This function performs all passes within a single transaction, committing only once.
    """
    total_flips = 0
    final_surface_id = initial_surface_id
    new_lengths_for_commit: Dict[HalfEdgeId, float] = {}

    with factory.update_topology([initial_surface_id]) as surgery:
        wb = surgery.workbench

        # Shim for adapting workbench to score functions that expect viewer-like proxies.
        shim_cache: Dict[HalfEdge, "_WBShim"] = {}

        def get_shim(he, lengths_map):
            if he is None:
                return None
            if he in shim_cache:
                return shim_cache[he]
            new_shim = _WBShim(he, lengths_map)
            shim_cache[he] = new_shim
            return new_shim

        class _WBShim:
            def __init__(self, he, lengths_map):
                self._he = he
                self._lengths = lengths_map

            @property
            def length(self):
                return self._lengths.get(self._he)

            @property
            def prev(self):
                return get_shim(self._he.prev, self._lengths)

            @property
            def next(self):
                return get_shim(self._he.next, self._lengths)

            @property
            def twin(self):
                return get_shim(self._he.twin, self._lengths)

            @property
            def origin(self):
                return self._he.origin

        # 1. Initialize a mutable dictionary for lengths on the workbench.
        wb_lengths = {}
        initial_view = factory.create_viewer(initial_surface_id)
        parent_id_to_wb_he = {
            he.parent_ids[0]: he for he in wb.half_edges if he.parent_ids
        }
        for he_proxy in initial_view.half_edges:
            if he_proxy.id in parent_id_to_wb_he:
                wb_he = parent_id_to_wb_he[he_proxy.id]
                try:
                    if he_proxy.length is not None:
                        wb_lengths[wb_he] = he_proxy.length
                except AttributeError:
                    # This is fine, it just means the length property was never set.
                    pass

        visited_states = set()
        if max_passes is None:
            max_passes = len(wb.half_edges) * 5 + 100

        for i in range(max_passes):
            current_edge_set = _get_canonical_edge_set_from_workbench(wb)
            if current_edge_set in visited_states:
                logging.warning("Cycle detected in refinement at pass %d. Stopping refinement gracefully.", i)
                break
            visited_states.add(current_edge_set)

            # Find all illegal edges on the current workbench state
            candidates = []
            for wb_he in wb.half_edges:
                if wb_he.is_deleted or not wb_he.twin or wb_he.twin.is_deleted:
                    continue
                shim = get_shim(wb_he, wb_lengths)
                score = score_fn(shim)
                if score > epsilon:
                    candidates.append((score, wb_he))

            if not candidates:
                logging.vlog(3, "Refinement stable after %d passes.", i)
                break
            candidates.sort(key=lambda x: x[0], reverse=True)

            flips_in_pass = 0

            # --- Pass 1: Try to flip without creating parallel edges ---
            for score, wb_he in candidates:
                if _can_flip_on_workbench(wb_he, allow_parallel_edges=False):
                    shim = get_shim(wb_he, wb_lengths)
                    new_len = length_calculation_fn(shim)
                    new_lengths_for_commit[wb_he.parent_ids[0]] = new_len

                    new_he, new_twin = _flip_edge_in_workbench(wb, wb_he)
                    wb_lengths[new_he] = new_len
                    wb_lengths[new_twin] = new_len
                    flips_in_pass += 1

            # --- Pass 2: If needed, try remaining candidates allowing parallel edges ---
            if flips_in_pass == 0:
                logging.vlog(
                    1,
                    "No safe flips found. Forcing flips that may create parallel edges.",
                )
                for score, wb_he in candidates:
                    if _can_flip_on_workbench(wb_he, allow_parallel_edges=True):
                        shim = get_shim(wb_he, wb_lengths)
                        new_len = length_calculation_fn(shim)
                        new_lengths_for_commit[wb_he.parent_ids[0]] = new_len

                        new_he, new_twin = _flip_edge_in_workbench(wb, wb_he)
                        wb_lengths[new_he] = new_len
                        wb_lengths[new_twin] = new_len
                        flips_in_pass += 1

            if flips_in_pass == 0:
                blocked_candidates = [c[1].parent_ids[0] for c in candidates]
                logging.error(
                    "Refinement stalled: all %d illegal edges are topologically blocked from flipping. Blocked candidates: %s.",
                    len(candidates),
                    blocked_candidates,
                )
                break
            total_flips += flips_in_pass
        else:
            logging.warning("Refinement did not converge after %d passes.", i)

        if total_flips > 0:
            final_surface_id = surgery.commit(f"Refined mesh with {total_flips} flips")

    # --- Annotate all new lengths in a single final transaction ---
    if total_flips > 0:
        with factory.update_properties(final_surface_id) as annotation:
            with annotation.writer() as new_view:
                parent_to_he_map = {
                    p: he for he in new_view.half_edges for p in he.parent_ids
                }
                for old_he_id, new_len in new_lengths_for_commit.items():
                    new_he = parent_to_he_map.get(old_he_id)
                    if new_he:
                        new_he.length = new_len
                        if new_he.twin:
                            new_he.twin.length = new_len
            final_surface_id = annotation.commit("Annotate lengths after refinement")

    return final_surface_id


# ==============================================================================
# --- Public-Facing API ---
# ==============================================================================


def refine_euclidean_delaunay(
    factory: SurfaceFactory, surface_id: SurfaceId, max_passes: int = None
) -> SurfaceId:
    """
    Refines the mesh to be Intrinsic Delaunay with respect to the Euclidean metric.
    Uses Euclidean law of cosines for flips.
    Appropriate for PREPROCESSING a Euclidean mesh before optimization.
    """
    return _refine(
        factory,
        surface_id,
        calculate_euclidean_flip_score,
        calculate_euclidean_flip_length,
        1e-8,
        max_passes,
    )


def refine_hyperbolic_delaunay(
    factory: SurfaceFactory, surface_id: SurfaceId, max_passes: int = None
) -> SurfaceId:
    """
    Refines the mesh to be Intrinsic Delaunay with respect to the hyperbolic metric.
    Uses hyperbolic law of cosines for flips.
    Appropriate for usage INSIDE the hyperbolic optimization loop.
    """
    return _refine(
        factory,
        surface_id,
        calculate_hyperbolic_flip_score,
        calculate_hyperbolic_flip_length,
        1e-8,
        max_passes,
    )


def refine_ceps_ptolemy(
    factory: SurfaceFactory, surface_id: SurfaceId, max_passes: int = None
) -> SurfaceId:
    """
    Refines the mesh to be Ideal Delaunay using Ptolemy flips.
    Appropriate for usage INSIDE the CEPS optimization loop where lengths represent
    exponentiated Penner coordinates.
    """
    # Eq 10 score is used for ALL refinement types per CEPS theory.
    return _refine(
        factory,
        surface_id,
        _calculate_ideal_flip_score,
        _calculate_ptolemy_flip_length,
        1e-8,
        max_passes,
    )
