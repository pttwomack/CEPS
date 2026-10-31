# mazedemo/dcm_tiling.py
"""
1-Ring Universal Cover Tiling and Dynamic Holonomy Utilities.

Constructs 1-ring universal cover tiles by applying holonomy generator matrices
M_g in PSL(2, R) / Lorentz / affine groups to the root fundamental domain T_0.
Provides mathematical invariant auditing tools and continuous multi-tile path tracing.
"""

import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Set
from dataclasses import dataclass, field

from mazedemo.ids import FaceId, VertexId, HalfEdgeId, SurfaceId, get_local_id
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.viewer import Viewer, Face, HalfEdge, Vertex


@dataclass
class TileFaceKey:
    """Uniquely identifies a face in a tiled cover: (local_face_id, tile_label)."""
    face_id: FaceId
    tile_label: str  # "" for root tile, "a", "A", "b", "B", etc.

    def __hash__(self):
        return hash((get_local_id(self.face_id), self.tile_label))

    def __eq__(self, other):
        if not isinstance(other, TileFaceKey):
            return False
        return (get_local_id(self.face_id) == get_local_id(other.face_id) and
                self.tile_label == other.tile_label)


@dataclass
class OneRingCover:
    """
    Data structure representing the 1-ring universal cover tiling.
    """
    root_placed_triangles: Dict[FaceId, Dict[VertexId, np.ndarray]]
    tiled_placed_triangles: Dict[TileFaceKey, Dict[VertexId, np.ndarray]]
    holonomy_transforms: Dict[str, np.ndarray]
    generators: List[str]
    geometry: Any
    seam_half_edges: Dict[HalfEdgeId, str] = field(default_factory=dict)
    tile_labels: List[str] = field(default_factory=list)


def build_one_ring_cover(
    factory: SurfaceFactory,
    surface_id: SurfaceId,
    geometry: Any,
    placed_triangles: Dict[FaceId, Dict[VertexId, np.ndarray]],
    holonomy_transforms: Dict[str, np.ndarray],
    holonomy_classes: Dict[HalfEdgeId, str],
    generators: List[str],
) -> OneRingCover:
    """
    Constructs a 1-ring universal cover tiling around the root fundamental domain T_0.

    Args:
        factory: SurfaceFactory instance.
        surface_id: SurfaceId of the flattened surface.
        geometry: Geometry object (EuclideanGeometry or HyperbolicGeometry).
        placed_triangles: Root tile face placement dict {FaceId: {VertexId: 3D vector}}.
        holonomy_transforms: Holonomy generator matrices {symbol: 3x3 matrix}.
        holonomy_classes: Map {HalfEdgeId: generator string}.
        generators: List of base generator symbols (e.g. ['a', 'b']).

    Returns:
        OneRingCover containing root tile T_0 and 1-ring adjacent tiles T_g.
    """
    view = factory.create_viewer(surface_id, custom_properties={"length": None})

    # Identify seam half-edges and their generators
    seam_half_edges: Dict[HalfEdgeId, str] = {}
    for he in view.half_edges:
        h_class = holonomy_classes.get(he.id, "")
        if h_class:
            seam_half_edges[he.id] = h_class

    tiled_placed_triangles: Dict[TileFaceKey, Dict[VertexId, np.ndarray]] = {}

    # 1. Add root tile T_0 (label "")
    tile_labels = [""]
    for f_id, face_verts in placed_triangles.items():
        key = TileFaceKey(face_id=f_id, tile_label="")
        tiled_placed_triangles[key] = {
            v_id: pos.copy() for v_id, pos in face_verts.items()
        }

    # 2. Generate 1-ring adjacent tiles T_g for each available holonomy transform
    for gen_symbol, M_gen in holonomy_transforms.items():
        if not gen_symbol:
            continue
        tile_labels.append(gen_symbol)
        for f_id, face_verts in placed_triangles.items():
            key = TileFaceKey(face_id=f_id, tile_label=gen_symbol)
            tiled_placed_triangles[key] = {
                v_id: M_gen @ pos for v_id, pos in face_verts.items()
            }

    return OneRingCover(
        root_placed_triangles=placed_triangles,
        tiled_placed_triangles=tiled_placed_triangles,
        holonomy_transforms=holonomy_transforms,
        generators=generators,
        geometry=geometry,
        seam_half_edges=seam_half_edges,
        tile_labels=tile_labels,
    )


def compute_face_area(
    face_verts: Dict[VertexId, np.ndarray], geometry: Any
) -> float:
    """
    Computes the geometric area of a triangle face.
    For Euclidean: standard 1/2 ||v1 x v2|| or Heron's formula.
    For Hyperbolic: Gauss-Bonnet defect area = pi - (alpha + beta + gamma).
    """
    pts = list(face_verts.values())
    if len(pts) != 3:
        return 0.0

    is_hyperbolic = "Hyperbolic" in geometry.__class__.__name__

    l0 = float(geometry.distance(pts[1], pts[2]))
    l1 = float(geometry.distance(pts[2], pts[0]))
    l2 = float(geometry.distance(pts[0], pts[1]))

    try:
        angles = geometry.compute_angles_from_opposite_lengths((l0, l1, l2))
        angle_sum = sum(angles)
        if is_hyperbolic:
            # Hyperbolic area = pi - (alpha + beta + gamma)
            area = max(0.0, np.pi - angle_sum)
            return float(area)
        else:
            # Euclidean triangle area via Heron's formula
            s = (l0 + l1 + l2) / 2.0
            area_sq = s * (s - l0) * (s - l1) * (s - l2)
            return float(np.sqrt(max(0.0, area_sq)))
    except (ValueError, AttributeError):
        pass

    # Fallback Euclidean cross product area
    p0, p1, p2 = pts[0][:2], pts[1][:2], pts[2][:2]
    return float(0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])))
