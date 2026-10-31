# mazedemo/vis.py
import collections
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from typing import Optional, Dict, Any, List
import networkx as nx

from mazedemo.viewer import Viewer
from mazedemo.hyperbolic import (
    HyperbolicGeometry,
    hyperboloid_to_poincare,
    is_on_hyperboloid,
)
from mazedemo.ids import VertexId, FaceId, get_local_id
from mazedemo.layout import to_cartesian


def _get_face_normal(points):
    """Calculates the normal vector of a face defined by three points."""
    if len(points) < 3:
        return np.array([0, 0, 1])
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else np.array([0, 0, 1])


def _get_rotation_to_z_axis(vec):
    """Calculates the rotation matrix that aligns a given vector with the z-axis."""
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return Rotation.identity()
    vec = vec / vec_norm
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(vec, z_axis)
    angle = np.arccos(np.dot(vec, z_axis))
    if np.linalg.norm(axis) < 1e-9:
        return Rotation.identity()
    return Rotation.from_rotvec(angle * (axis / np.linalg.norm(axis)))


def _stereographic_projection(points_3d):
    """Projects 3D points from a sphere onto a 2D plane."""
    projected = []
    for p in points_3d:
        x, y, z = p
        if np.isclose(z, 1.0):
            projected.append([np.inf, np.inf])
        else:
            projected.append([x / (1 - z), y / (1 - z)])
    return np.array(projected)


def plot_dcel(
    view: Viewer,
    title: str = "Surface_Visualization",
    explode_face_id: Optional[int] = None,
):
    """Generates and saves a detailed plot of a surface's topology using a Viewer."""
    ARROW_MARGIN = 0.15
    ARROW_HEAD_WIDTH = 0.05
    ARROW_LINE_WIDTH = 0.5
    VERTEX_MARKER_SIZE = 8
    VERTEX_LABEL_OFFSET = 0.05
    HE_LABEL_FONTSIZE = 6
    VERTEX_LABEL_FONTSIZE = 8
    FACE_LABEL_FONTSIZE = 10
    FACE_LABEL_PAD = 0.3

    fig, ax = plt.subplots(figsize=(12, 12))
    projected_coords: Dict[VertexId, np.ndarray] = {}

    first_vertex_pos = view.get_property(view.vertices[0].id, "pos")
    if first_vertex_pos is not None and len(first_vertex_pos) == 3:
        all_points_3d = np.array([view.get_property(v.id, "pos") for v in view.vertices])
        rotation = Rotation.identity()

        if explode_face_id is not None:
            face_to_explode_id = FaceId(view._surface_id, explode_face_id)
            face_to_explode = view.get_proxy(face_to_explode_id)

            if face_to_explode:
                face_points = np.array(
                    [
                        view.get_property(he.origin.id, "pos")
                        for he in face_to_explode.half_edges
                    ]
                )
                face_normal = _get_face_normal(face_points)
                rotation = _get_rotation_to_z_axis(face_normal)

        all_points_3d = rotation.apply(all_points_3d)

        norms = np.linalg.norm(all_points_3d, axis=1, keepdims=True)
        norms[norms == 0] = 1
        all_points_3d /= norms

        projected_points_2d = _stereographic_projection(all_points_3d)

        for v, p2d in zip(view.vertices, projected_points_2d):
            projected_coords[v.id] = p2d
    else:
        for v in view.vertices:
            projected_coords[v.id] = view.get_property(v.id, "pos")

    for i, he in enumerate(view.half_edges):
        p1 = projected_coords.get(he.origin.id)
        p2 = (
            projected_coords.get(he.twin.origin.id)
            if he.twin
            else projected_coords.get(he.next.origin.id)
        )

        if p1 is None or p2 is None:
            continue

        edge_color = "lightcoral" if he.twin is None else "lightblue"
        text_color = "darkred" if he.twin is None else "blue"

        vector = p2 - p1
        length = np.linalg.norm(vector)
        if length > 0 and np.all(np.isfinite([p1, p2])):
            direction = vector / length

            if length > 2 * ARROW_MARGIN:
                start = p1 + direction * ARROW_MARGIN
                arrow_vec = direction * (length - 2 * ARROW_MARGIN)
                ax.arrow(
                    start[0],
                    start[1],
                    arrow_vec[0],
                    arrow_vec[1],
                    head_width=ARROW_HEAD_WIDTH,
                    length_includes_head=True,
                    fc=edge_color,
                    ec="black",
                    lw=ARROW_LINE_WIDTH,
                    zorder=1,
                )

        if np.all(np.isfinite([p1, p2])):
            midpoint = p1 + vector * 0.5
            label_pos = midpoint

            if he.face:
                face_verts_pos = [
                    projected_coords.get(he_in_face.origin.id)
                    for he_in_face in he.face.half_edges
                ]
                valid_face_verts = [
                    pos
                    for pos in face_verts_pos
                    if pos is not None and np.all(np.isfinite(pos))
                ]

                if valid_face_verts:
                    face_centroid = np.mean(valid_face_verts, axis=0)
                    if np.all(np.isfinite(face_centroid)):
                        offset_vec = face_centroid - midpoint
                        offset_norm = np.linalg.norm(offset_vec)
                        if offset_norm > 0:
                            label_pos = midpoint + 0.1 * (offset_vec / offset_norm)

            ax.text(
                label_pos[0],
                label_pos[1],
                f"he_{i}",
                fontsize=HE_LABEL_FONTSIZE,
                color=text_color,
                ha="center",
                va="center",
                zorder=3,
            )

    for v in view.vertices:
        pos = projected_coords.get(v.id)
        if pos is not None and np.all(np.isfinite(pos)):
            ax.plot(
                pos[0],
                pos[1],
                "o",
                markersize=VERTEX_MARKER_SIZE,
                color="red",
                zorder=2,
            )
            ax.text(
                pos[0] + VERTEX_LABEL_OFFSET,
                pos[1] + VERTEX_LABEL_OFFSET,
                str(v.id),
                fontsize=VERTEX_LABEL_FONTSIZE,
                color="black",
                ha="left",
                zorder=3,
            )

    for f in view.faces:
        face_verts_pos = [projected_coords.get(he.origin.id) for he in f.half_edges]
        valid_pos = [
            pos
            for pos in face_verts_pos
            if pos is not None and np.all(np.isfinite(pos))
        ]

        if valid_pos:
            centroid = np.mean(valid_pos, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(f.id),
                fontsize=FACE_LABEL_FONTSIZE,
                color="darkgreen",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle=f"circle,pad={FACE_LABEL_PAD}",
                    fc="lightgreen",
                    ec="g",
                    lw=1,
                    alpha=0.9,
                ),
                zorder=4,
            )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.1)
    plt.tight_layout()

    all_pos = np.array(list(projected_coords.values()))
    if all_pos.size > 0:
        finite_pos = all_pos[np.all(np.isfinite(all_pos), axis=1)]
        if finite_pos.size > 0:
            ax.set_xlim(np.min(finite_pos[:, 0]) - 0.5, np.max(finite_pos[:, 0]) + 0.5)
            ax.set_ylim(np.min(finite_pos[:, 1]) - 0.5, np.max(finite_pos[:, 1]) + 0.5)
    plt.close()


def _get_geodesic_arc_params(p1: np.ndarray, p2: np.ndarray) -> tuple:
    if np.abs(p1[0] * p2[1] - p1[1] * p2[0]) < 1e-9:
        return (True, p1, p2, None, None)

    A = np.array([[p2[0] - p1[0], p2[1] - p1[1]], [p1[0], p1[1]]])
    b = np.array([(np.dot(p2, p2) - np.dot(p1, p1)) / 2.0, (1 + np.dot(p1, p1)) / 2.0])

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return (True, p1, p2, None, None)

    radius = np.sqrt(np.dot(center, center) - 1)
    start_angle = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    end_angle = np.arctan2(p2[1] - center[1], p2[0] - center[0])

    return (False, center, radius, start_angle, end_angle)


def _get_geodesic_path(
    p1: np.ndarray, p2: np.ndarray, num_samples: int = 30
) -> np.ndarray:
    is_line, center, radius, start_angle, end_angle = _get_geodesic_arc_params(p1, p2)
    if is_line:
        return np.array([p1, p2])

    delta_angle = end_angle - start_angle
    if delta_angle > np.pi:
        delta_angle -= 2 * np.pi
    elif delta_angle < -np.pi:
        delta_angle += 2 * np.pi

    thetas = np.linspace(start_angle, start_angle + delta_angle, num_samples)
    points = center[:, np.newaxis] + radius * np.array([np.cos(thetas), np.sin(thetas)])
    return points.T


def plot_dilatation_heatmap(
    factory: Viewer,
    context: Any,
    placements: Optional[Dict[FaceId, Dict[VertexId, np.ndarray]]] = None,
    geometry: Any = None,
    title: str = "Local Dilatation Heatmap (K_f)",
    ax: Optional[plt.Axes] = None,
    bridge: Optional[Any] = None,
):
    """
    Renders Phase 3 Single-Panel Local Dilatation Heatmap (K_f).
    Uses CanonicalBridge under the hood to ensure exact numeric & visual 1:1 consistency
    with Phase 2 Dual-Panel plots and Phase 4 Telemetry Scorecards.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    from mazedemo.bridge import CanonicalBridge
    if bridge is None:
        bridge = CanonicalBridge(factory, context, geometry)

    # Calibrate color scale strictly on base surface faces
    view_base = bridge.factory.create_viewer(
        bridge.base_surface_id, custom_properties={"position": None, "pos": None}
    )
    base_local_ids = {get_local_id(f.id) for f in view_base.faces}
    base_vals = [
        float(val)
        for fid, val in bridge.local_dilatation_map.items()
        if get_local_id(fid) in base_local_ids
    ] or [1.0]

    v_min = 1.0
    v_max = max(1.5, float(np.max(base_vals)))
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cmap = cm.viridis
    is_hyperbolic = bridge.layout_metadata["is_hyperbolic"]

    all_2d_pts = []

    for f_id, placement in bridge.placed_triangles.items():
        local_f_id = get_local_id(f_id)
        K_f = bridge.local_dilatation_map.get(f_id, bridge.local_dilatation_map.get(local_f_id, 1.0))
        face_color = cmap(norm(min(max(K_f, v_min), v_max)))

        verts = list(placement.keys())
        p0, p1, p2 = placement[verts[0]], placement[verts[1]], placement[verts[2]]

        if is_hyperbolic:
            cp0, cp1, cp2 = hyperboloid_to_poincare(p0), hyperboloid_to_poincare(p1), hyperboloid_to_poincare(p2)
            path1 = _get_geodesic_path(cp0, cp1)
            path2 = _get_geodesic_path(cp1, cp2)
            path3 = _get_geodesic_path(cp2, cp0)
            full_path = np.concatenate([path1, path2, path3])
            polygon = plt.Polygon(
                full_path, closed=True, facecolor=face_color, edgecolor="k", linewidth=0.5, alpha=0.85
            )
            all_2d_pts.extend([cp0, cp1, cp2])
        else:
            cp0, cp1, cp2 = to_cartesian(p0), to_cartesian(p1), to_cartesian(p2)
            polygon = plt.Polygon(
                [cp0, cp1, cp2], closed=True, facecolor=face_color, edgecolor="k", linewidth=0.5, alpha=0.85
            )
            all_2d_pts.extend([cp0, cp1, cp2])

        ax.add_patch(polygon)

    if is_hyperbolic:
        disk_boundary = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
        ax.add_artist(disk_boundary)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        if all_2d_pts:
            pts_arr = np.array(all_2d_pts)
            xmin, xmax = np.min(pts_arr[:, 0]), np.max(pts_arr[:, 0])
            ymin, ymax = np.min(pts_arr[:, 1]), np.max(pts_arr[:, 1])
            pad_x = max(0.1, 0.05 * (xmax - xmin))
            pad_y = max(0.1, 0.05 * (ymax - ymin))
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Local Dilatation Ratio K_f (1.0 = Conformal/Isometric)")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.1)


def plot_canonical_dual_panel(
    bridge: Any,
    path_result: Optional[Any] = None,
    g_grid: Optional[nx.Graph] = None,
    show_labels: bool = True,
    show_seams: bool = True,
    show_walls: bool = True,
    scalar_field: str = "K_f",
    title: str = "Module 3: Dual-Panel Canonical Bridge Analysis",
) -> pd.DataFrame:
    """
    Renders high-resolution dual-panel visualization for Module 3:
    - Left Panel: Renders strictly pristine pre-IDT base grid faces from bridge.base_surface_id.
    - Right Panel: Canonical fundamental domain tile.
    - Color Scale Calibration: Calibrated strictly on base surface faces so high K_f
      outlier refinement faces never blow out the global color gradient.
    - Legend: Reports base faces with matching hex pass, and marks IDT refinement/seam
      faces as 'N/A (Refinement/Seam Face)'.
    """
    fig, (ax_grid, ax_domain) = plt.subplots(1, 2, figsize=(16, 7))

    view_base = bridge.factory.create_viewer(
        bridge.base_surface_id, custom_properties={"position": None, "pos": None}
    )
    base_local_ids = {get_local_id(f.id) for f in view_base.faces}

    scalar_by_local_id: Dict[int, float] = {}
    for fid, val in bridge.local_dilatation_map.items():
        local_id = get_local_id(fid)
        scalar_by_local_id[local_id] = float(val)

    if scalar_field.lower() == "u":
        u_arr = bridge.context.u
        view_solver = bridge.factory.create_viewer(
            bridge.surface_id, custom_properties={"dcm_index": None}
        )
        scalar_by_local_id = {}
        for face in view_solver.faces:
            local_id = get_local_id(face.id)
            u_vals = [
                float(u_arr[he.origin.dcm_index])
                for he in face.half_edges
                if he.origin.dcm_index is not None and he.origin.dcm_index < len(u_arr)
            ]
            scalar_by_local_id[local_id] = float(np.mean(u_vals)) if u_vals else 0.0

    # Calibrate color normalization strictly on base surface faces
    base_raw_vals = [
        scalar_by_local_id[lid] for lid in base_local_ids if lid in scalar_by_local_id
    ] or [1.0]

    if scalar_field.lower() == "u":
        cmap = cm.magma
        v_min, v_max = float(np.min(base_raw_vals)), float(np.max(base_raw_vals))
        if np.isclose(v_min, v_max):
            v_min, v_max = v_min - 0.5, v_max + 0.5
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        colorbar_label = "Scale Factor u_i (Log Metric Scaling)"
    else:
        cmap = cm.viridis
        v_min = 1.0
        v_max = max(1.5, float(np.max(base_raw_vals)))
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        colorbar_label = "Local Dilatation Ratio K_f (1.0 = Conformal/Isometric)"

    grid_pts = []
    domain_pts = []
    is_hyperbolic = bridge.layout_metadata["is_hyperbolic"]
    holonomy_classes = bridge.layout_metadata.get("holonomy_classes", {})

    left_face_colors: Dict[int, str] = {}
    right_face_colors: Dict[int, str] = {}

    # 1. Left Panel: Render all faces present in view_base (pristine pre-IDT base surface mesh)
    for face in view_base.faces:
        local_f_id = get_local_id(face.id)

        val = scalar_by_local_id.get(local_f_id, 1.0)
        color_rgba = cmap(norm(min(max(val, v_min), v_max)))
        hex_color = mcolors.to_hex(color_rgba)

        v_positions = []
        for he in face.half_edges:
            pos = view_base.get_property(he.origin.id, "position")
            if pos is None:
                pos = view_base.get_property(he.origin.id, "pos")
            if pos is not None and isinstance(pos, (tuple, list, np.ndarray)) and len(pos) >= 2:
                v_positions.append(np.array(pos[:2], dtype=np.float64))

        if len(v_positions) == 3:
            left_face_colors[local_f_id] = hex_color
            poly_grid = plt.Polygon(
                v_positions,
                closed=True,
                facecolor=color_rgba,
                edgecolor="w",
                linewidth=0.6,
                alpha=0.85,
            )
            ax_grid.add_patch(poly_grid)
            grid_pts.extend(v_positions)

            if show_labels:
                centroid_grid = np.mean(v_positions, axis=0)
                ax_grid.text(
                    centroid_grid[0],
                    centroid_grid[1],
                    f"F{local_f_id}",
                    fontsize=7,
                    color="lightgreen",
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=5,
                )

    # 2. Right Panel: Render fundamental domain placement triangles
    for f_id, placement in bridge.placed_triangles.items():
        local_f_id = get_local_id(f_id)
        val = scalar_by_local_id.get(local_f_id, 1.0)
        color_rgba = cmap(norm(min(max(val, v_min), v_max)))
        hex_color = mcolors.to_hex(color_rgba)
        right_face_colors[local_f_id] = hex_color

        verts = list(placement.keys())
        p0, p1, p2 = placement[verts[0]], placement[verts[1]], placement[verts[2]]

        if is_hyperbolic:
            # PROJECT VERTICES TO 2D POINCARÉ DISK FIRST BEFORE COMPUTING CENTROIDS
            cp0, cp1, cp2 = hyperboloid_to_poincare(p0), hyperboloid_to_poincare(p1), hyperboloid_to_poincare(p2)
            path1 = _get_geodesic_path(cp0, cp1)
            path2 = _get_geodesic_path(cp1, cp2)
            path3 = _get_geodesic_path(cp2, cp0)
            full_path = np.concatenate([path1, path2, path3])
            polygon_dom = plt.Polygon(
                full_path, closed=True, facecolor=color_rgba, edgecolor="k", linewidth=0.5, alpha=0.85
            )
            domain_pts.extend([cp0, cp1, cp2])
            centroid_dom = np.mean([cp0, cp1, cp2], axis=0)
        else:
            cp0, cp1, cp2 = to_cartesian(p0), to_cartesian(p1), to_cartesian(p2)
            polygon_dom = plt.Polygon(
                [cp0, cp1, cp2], closed=True, facecolor=color_rgba, edgecolor="k", linewidth=0.5, alpha=0.85
            )
            domain_pts.extend([cp0, cp1, cp2])
            centroid_dom = np.mean([cp0, cp1, cp2], axis=0)

        ax_domain.add_patch(polygon_dom)

        if show_labels:
            ax_domain.text(
                centroid_dom[0],
                centroid_dom[1],
                f"F{local_f_id}",
                fontsize=7,
                color="lightgreen",
                ha="center",
                va="center",
                weight="bold",
                zorder=5,
            )

    if show_seams:
        for face in view_base.faces:
            for he in face.half_edges:
                h_class = holonomy_classes.get(he.id)
                if h_class:
                    p1 = view_base.get_property(he.origin.id, "position")
                    if p1 is None:
                        p1 = view_base.get_property(he.origin.id, "pos")

                    p2 = (
                        view_base.get_property(he.twin.origin.id, "position")
                        if he.twin
                        else view_base.get_property(he.next.origin.id, "position")
                    )
                    if p2 is None and he.twin:
                        p2 = view_base.get_property(he.twin.origin.id, "pos")
                    elif p2 is None:
                        p2 = view_base.get_property(he.next.origin.id, "pos")

                    if p1 is not None and p2 is not None and len(p1) >= 2 and len(p2) >= 2:
                        ax_grid.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            "r--",
                            lw=2.0,
                            zorder=4,
                        )

    if show_walls and bridge.maze_walls:
        for p1, p2 in bridge.maze_walls:
            ax_grid.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                "k-",
                lw=2.5,
                zorder=5,
            )

    if path_result is not None and hasattr(path_result, "grid_path_coords"):
        coords = path_result.grid_path_coords
        ax_grid.plot(coords[:, 0], coords[:, 1], "c-", lw=3.0, zorder=6, label="Continuous Geodesic")
        ax_grid.plot(coords[0, 0], coords[0, 1], "go", ms=8, zorder=7, label="Start")
        ax_grid.plot(coords[-1, 0], coords[-1, 1], "ro", ms=8, zorder=7, label="End")

    if grid_pts:
        grid_arr = np.array(grid_pts)
        pad_x = max(0.2, 0.05 * (np.max(grid_arr[:, 0]) - np.min(grid_arr[:, 0])))
        pad_y = max(0.2, 0.05 * (np.max(grid_arr[:, 1]) - np.min(grid_arr[:, 1])))
        ax_grid.set_xlim(np.min(grid_arr[:, 0]) - pad_x, np.max(grid_arr[:, 0]) + pad_x)
        ax_grid.set_ylim(np.min(grid_arr[:, 1]) - pad_y, np.max(grid_arr[:, 1]) + pad_y)

    ax_grid.set_title("1. Raw 2D Maze Grid Space", fontsize=11, fontweight="bold")
    ax_grid.set_aspect("equal", adjustable="box")

    if is_hyperbolic:
        disk_boundary = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
        ax_domain.add_artist(disk_boundary)
        ax_domain.set_xlim(-1.1, 1.1)
        ax_domain.set_ylim(-1.1, 1.1)
        ax_domain.set_title("2. Canonical Poincaré Disk D", fontsize=11, fontweight="bold")
    else:
        if domain_pts:
            d_arr = np.array(domain_pts)
            xmin, xmax = np.min(d_arr[:, 0]), np.max(d_arr[:, 0])
            ymin, ymax = np.min(d_arr[:, 1]), np.max(d_arr[:, 1])
            pad_x = max(0.1, 0.05 * (xmax - xmin))
            pad_y = max(0.1, 0.05 * (ymax - ymin))
            ax_domain.set_xlim(xmin - pad_x, xmax + pad_x)
            ax_domain.set_ylim(ymin - pad_y, ymax + pad_y)
        ax_domain.set_title("2. Canonical Euclidean Annulus", fontsize=11, fontweight="bold")

    ax_domain.set_aspect("equal", adjustable="box")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_grid, label=colorbar_label)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 3. Legend Match Table
    table_rows = []
    all_local_ids = sorted(set(left_face_colors.keys()).union(set(right_face_colors.keys())))

    for local_id in all_local_ids:
        val = scalar_by_local_id.get(local_id, 1.0)
        left_hex = left_face_colors.get(local_id, "N/A")
        right_hex = right_face_colors.get(local_id, "N/A")

        if local_id not in base_local_ids or left_hex == "N/A":
            match_status = "N/A (Refinement/Seam Face)"
        elif right_hex == "N/A":
            match_status = "NO (Missing in Domain Layout)"
        elif left_hex == right_hex:
            match_status = "YES"
        else:
            match_status = "NO"

        table_rows.append({
            "Face ID": f"F{local_id}",
            f"Numeric Value ({scalar_field})": f"{val:.4f}",
            "Left Panel Hex Color": left_hex,
            "Right Panel Hex Color": right_hex,
            "Color Match Pass": match_status,
        })

    df_legend = pd.DataFrame(table_rows)

    try:
        from IPython.display import display
        display(df_legend)
    except Exception:
        pass

    return df_legend


def plot_one_ring_cover(
    bridge: Any,
    path_result: Optional[Any] = None,
    title: str = "1-Ring Universal Cover Tiling & Continuous Pathfinder",
):
    """
    Renders Topology & Pathfinder 2-Panel Plot:
    - Left Panel: Clean 1-Ring Universal Cover Tiling with clear bold cut seam tile boundaries,
      light internal triangle edges, and prominent red continuous geodesic path crossing seam boundaries.
    - Right Panel: 2D Maze Grid with continuous 2D pathfinder (zero teleportation).
    - Separation of Concerns: Face background fill is kept clean and translucent (alpha=0.15) to make the
      red path and bold tile boundaries the clear visual priority.
    """
    fig, (ax_cover, ax_maze) = plt.subplots(1, 2, figsize=(16, 7))

    cover = bridge.one_ring_cover
    is_hyperbolic = bridge.layout_metadata["is_hyperbolic"]

    tile_boundary_colors = ["#1e88e5", "#e53935", "#43a047", "#fdd835", "#8e24aa", "#3949ab"]
    tile_color_map = {}
    for idx, label in enumerate(cover.tile_labels):
        tile_color_map[label] = tile_boundary_colors[idx % len(tile_boundary_colors)]

    cover_pts = []

    # 1. Render faces in 1-ring cover with clean, light background and distinct tile boundary outlines
    for key, face_verts in cover.tiled_placed_triangles.items():
        is_root = (key.tile_label == "")
        tile_edge_color = tile_color_map.get(key.tile_label, "black")

        v_ids = list(face_verts.keys())
        p0, p1, p2 = face_verts[v_ids[0]], face_verts[v_ids[1]], face_verts[v_ids[2]]

        if is_hyperbolic:
            cp0, cp1, cp2 = hyperboloid_to_poincare(p0), hyperboloid_to_poincare(p1), hyperboloid_to_poincare(p2)
            path1 = _get_geodesic_path(cp0, cp1)
            path2 = _get_geodesic_path(cp1, cp2)
            path3 = _get_geodesic_path(cp2, cp0)
            full_path = np.concatenate([path1, path2, path3])
            polygon = plt.Polygon(
                full_path, closed=True, facecolor="#f0f4c3" if is_root else "#f5f5f5",
                edgecolor=tile_edge_color if is_root else "#bdbdbd", linewidth=1.2 if is_root else 0.5, alpha=0.35 if is_root else 0.15
            )
            cover_pts.extend([cp0, cp1, cp2])
        else:
            cp0, cp1, cp2 = p0[:2], p1[:2], p2[:2]
            polygon = plt.Polygon(
                [cp0, cp1, cp2], closed=True, facecolor="#f0f4c3" if is_root else "#f5f5f5",
                edgecolor=tile_edge_color if is_root else "#bdbdbd", linewidth=1.2 if is_root else 0.5, alpha=0.35 if is_root else 0.15
            )
            cover_pts.extend([cp0, cp1, cp2])

        ax_cover.add_patch(polygon)

    # 2. Draw bold tile boundary cut seam outlines for 1-ring adjacent tiles
    tile_groups = collections.defaultdict(list)
    for key, face_verts in cover.tiled_placed_triangles.items():
        tile_groups[key.tile_label].append(face_verts)

    for label, faces in tile_groups.items():
        b_color = tile_color_map.get(label, "black")
        for face_verts in faces:
            v_ids = list(face_verts.keys())
            pts = [face_verts[vid] for vid in v_ids]
            if is_hyperbolic:
                pts_2d = [hyperboloid_to_poincare(pt) for pt in pts]
                p0, p1, p2 = pts_2d[0], pts_2d[1], pts_2d[2]
                path1 = _get_geodesic_path(p0, p1)
                path2 = _get_geodesic_path(p1, p2)
                path3 = _get_geodesic_path(p2, p0)
                ax_cover.plot(path1[:, 0], path1[:, 1], color=b_color, lw=1.2 if label == "" else 0.8, alpha=0.8)
                ax_cover.plot(path2[:, 0], path2[:, 1], color=b_color, lw=1.2 if label == "" else 0.8, alpha=0.8)
                ax_cover.plot(path3[:, 0], path3[:, 1], color=b_color, lw=1.2 if label == "" else 0.8, alpha=0.8)
            else:
                pts_2d = [pt[:2] for pt in pts]
                pts_loop = np.array(pts_2d + [pts_2d[0]])
                ax_cover.plot(pts_loop[:, 0], pts_loop[:, 1], color=b_color, lw=1.2 if label == "" else 0.8, alpha=0.8)

    # 3. Render prominent continuous geodesic path
    if path_result is not None and hasattr(path_result, "poincare_arc_samples"):
        arc = path_result.poincare_arc_samples
        if isinstance(arc[0], (complex, np.complexfloating)):
            arc_pts = np.column_stack([np.real(arc), np.imag(arc)])
        else:
            arc_pts = arc
        ax_cover.plot(arc_pts[:, 0], arc_pts[:, 1], "r-", lw=3.2, zorder=8, label="Continuous Geodesic Arc")
        ax_cover.plot(arc_pts[0, 0], arc_pts[0, 1], "go", ms=9, zorder=9, label="Start")
        ax_cover.plot(arc_pts[-1, 0], arc_pts[-1, 1], "ro", ms=9, zorder=9, label="End")

    if is_hyperbolic:
        disk_boundary = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
        ax_cover.add_artist(disk_boundary)
        ax_cover.set_xlim(-1.1, 1.1)
        ax_cover.set_ylim(-1.1, 1.1)
        ax_cover.set_title("1. 1-Ring Universal Cover Tiling (Bold Cut Seam Boundaries)", fontsize=11, fontweight="bold")
    else:
        if cover_pts:
            pts_arr = np.array(cover_pts)
            xmin, xmax = np.min(pts_arr[:, 0]), np.max(pts_arr[:, 0])
            ymin, ymax = np.min(pts_arr[:, 1]), np.max(pts_arr[:, 1])
            pad_x = max(0.2, 0.05 * (xmax - xmin))
            pad_y = max(0.2, 0.05 * (ymax - ymin))
            ax_cover.set_xlim(xmin - pad_x, xmax + pad_x)
            ax_cover.set_ylim(ymin - pad_y, ymax + pad_y)
        ax_cover.set_title("1. 1-Ring Universal Cover Tiling (Euclidean Space)", fontsize=11, fontweight="bold")

    ax_cover.set_aspect("equal", adjustable="box")
    ax_cover.legend(loc="upper right", fontsize=8)

    # 4. Render Right Panel: 2D Maze Grid with continuous path
    view_base = bridge.factory.create_viewer(
        bridge.base_surface_id, custom_properties={"position": None, "pos": None}
    )
    base_local_ids = {get_local_id(f.id) for f in view_base.faces}
    grid_pts = []

    for face in view_base.faces:
        v_positions = []
        for he in face.half_edges:
            pos = view_base.get_property(he.origin.id, "position")
            if pos is None:
                pos = view_base.get_property(he.origin.id, "pos")
            if pos is not None and isinstance(pos, (tuple, list, np.ndarray)) and len(pos) >= 2:
                v_positions.append(np.array(pos[:2], dtype=np.float64))

        if len(v_positions) == 3:
            poly_grid = plt.Polygon(
                v_positions,
                closed=True,
                facecolor="#f5f5f5",
                edgecolor="#bdbdbd",
                linewidth=0.5,
                alpha=0.6,
            )
            ax_maze.add_patch(poly_grid)
            grid_pts.extend(v_positions)

    if bridge.maze_walls:
        for p1, p2 in bridge.maze_walls:
            ax_maze.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=2.5, zorder=5)

    if path_result is not None and hasattr(path_result, "grid_path_coords"):
        coords = path_result.grid_path_coords
        ax_maze.plot(coords[:, 0], coords[:, 1], "r-", lw=3.2, zorder=8, label="Continuous 2D Pathfinder")
        ax_maze.plot(coords[0, 0], coords[0, 1], "go", ms=9, zorder=9)
        ax_maze.plot(coords[-1, 0], coords[-1, 1], "ro", ms=9, zorder=9)

    if grid_pts:
        grid_arr = np.array(grid_pts)
        pad_x = max(0.2, 0.05 * (np.max(grid_arr[:, 0]) - np.min(grid_arr[:, 0])))
        pad_y = max(0.2, 0.05 * (np.max(grid_arr[:, 1]) - np.min(grid_arr[:, 1])))
        ax_maze.set_xlim(np.min(grid_arr[:, 0]) - pad_x, np.max(grid_arr[:, 0]) + pad_x)
        ax_maze.set_ylim(np.min(grid_arr[:, 1]) - pad_y, np.max(grid_arr[:, 1]) + pad_y)

    ax_maze.set_title("2. Continuous 2D Maze Pathfinder (Zero Teleportation)", fontsize=11, fontweight="bold")
    ax_maze.set_aspect("equal", adjustable="box")
    ax_maze.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_bidirectional_mapping(
    factory: Viewer,
    context: Any,
    placements: Dict[FaceId, Dict[VertexId, np.ndarray]],
    geometry: Any,
    title: str = "Two-Way Geometric Bridge (Maze Grid <-> Fundamental Domain)",
):
    """Legacy helper routing to plot_canonical_dual_panel."""
    from mazedemo.bridge import CanonicalBridge
    bridge = CanonicalBridge(factory, context, geometry)
    return plot_canonical_dual_panel(bridge, scalar_field="u", title=title)


def plot_fundamental_domain(
    placements: Dict[FaceId, Dict[VertexId, np.ndarray]],
    spanning_tree: nx.Graph,
    holonomy_transforms: Dict[str, np.ndarray],
    geometry: Any,
    title: str = "Fundamental_Domain",
    ax: Optional[plt.Axes] = None,
):
    """
    Mode 1 Visualization: Visualizes the laid-out fundamental domain tile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    if isinstance(geometry, HyperbolicGeometry):
        disk_boundary = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
        ax.add_artist(disk_boundary)

        projected_placements = {}
        for face_id, placement in placements.items():
            projected_placements[face_id] = {
                v_id: hyperboloid_to_poincare(pos) for v_id, pos in placement.items()
            }

        plot_placements = projected_placements
        is_hyperbolic = True
    else:
        plot_placements = {
            face_id: {v_id: pos[:2] for v_id, pos in placement.items()}
            for face_id, placement in placements.items()
        }
        is_hyperbolic = False

    plotted_locations = {}
    for face_id, placement in plot_placements.items():
        verts = list(placement.keys())
        p0, p1, p2 = placement[verts[0]], placement[verts[1]], placement[verts[2]]

        if is_hyperbolic:
            path1 = _get_geodesic_path(p0, p1)
            path2 = _get_geodesic_path(p1, p2)
            path3 = _get_geodesic_path(p2, p0)

            full_path = np.concatenate([path1, path2, path3])
            polygon = plt.Polygon(
                full_path, closed=True, facecolor="lightblue", alpha=0.6
            )
            ax.add_patch(polygon)

            ax.plot(path1[:, 0], path1[:, 1], "k-", lw=1)
            ax.plot(path2[:, 0], path2[:, 1], "k-", lw=1)
            ax.plot(path3[:, 0], path3[:, 1], "k-", lw=1)
        else:
            polygon = plt.Polygon(
                [p0, p1, p2],
                closed=True,
                edgecolor="k",
                facecolor="lightblue",
                alpha=0.6,
            )
            ax.add_patch(polygon)

        centroid = np.mean([p0, p1, p2], axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(face_id),
            fontsize=10,
            color="darkgreen",
            ha="center",
            va="center",
        )

        for vertex_id, pos in placement.items():
            pos_key = (round(pos[0], 4), round(pos[1], 4))
            if pos_key in plotted_locations:
                if plotted_locations[pos_key] != vertex_id:
                    ax.plot(
                        pos[0], pos[1], "o", markersize=15, mfc="none", mec="red", mew=2
                    )
                    ax.text(
                        pos[0],
                        pos[1] + 0.05,
                        f"{plotted_locations[pos_key]}/{vertex_id}",
                        fontsize=9,
                        color="red",
                        ha="center",
                        weight="bold",
                    )
            else:
                plotted_locations[pos_key] = vertex_id
                ax.text(
                    pos[0],
                    pos[1] + 0.03,
                    str(vertex_id),
                    fontsize=8,
                    color="dimgray",
                    ha="center",
                )

    face_centroids = {
        face_id: np.mean(np.array(list(placement.values())), axis=0)
        for face_id, placement in plot_placements.items()
    }

    for u, v in spanning_tree.edges():
        p1, p2 = face_centroids.get(u), face_centroids.get(v)
        if p1 is not None and p2 is not None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", lw=1.5)

    origin_hyperboloid = np.array([0.0, 0.0, 1.0])
    origin_poincare = np.array([0.0, 0.0])
    for name, transform in holonomy_transforms.items():
        if len(name) != 1:
            continue

        if is_hyperbolic:
            transformed_origin_h = transform @ origin_hyperboloid
            if is_on_hyperboloid(transformed_origin_h):
                vec = hyperboloid_to_poincare(transformed_origin_h)
                ax.arrow(
                    origin_poincare[0],
                    origin_poincare[1],
                    vec[0],
                    vec[1],
                    head_width=0.05,
                    length_includes_head=True,
                    fc="purple",
                    ec="purple",
                    lw=2,
                    zorder=5,
                )
                ax.text(
                    vec[0] * 1.1,
                    vec[1] * 1.1,
                    name,
                    color="purple",
                    fontsize=14,
                    weight="bold",
                )
        else:
            if transform.shape == (3, 3):
                vec = transform[:2, 2]
                ax.arrow(
                    origin_poincare[0],
                    origin_poincare[1],
                    vec[0],
                    vec[1],
                    head_width=0.2,
                    length_includes_head=True,
                    fc="purple",
                    ec="purple",
                    lw=2,
                    zorder=5,
                )
                ax.text(
                    vec[0] * 1.1,
                    vec[1] * 1.1,
                    name,
                    color="purple",
                    fontsize=14,
                    weight="bold",
                )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.1)
    plt.tight_layout()
    plt.show()
