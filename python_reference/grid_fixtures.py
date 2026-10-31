import numpy as np
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId


def _create_triangulated_grid(
    factory: SurfaceFactory,
    rows: int,
    cols: int,
    wrap_x: bool,
    wrap_y: bool,
    grid_type: str,
    perturb_amount: float = 0.0,
) -> SurfaceId:
    """
    The master private helper. Creates and annotates a surface for a general
    parallelogram grid, using a specified triangulation pattern and optional perturbation.
    """
    if grid_type == "square":
        v_translation = (0.0, 1.0)
        diagonal_pattern = "v00_v11"
    elif grid_type == "equilateral":
        v_translation = (0.5, np.sqrt(3) / 2.0)
        diagonal_pattern = "v01_v10"
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
    # --- 1. Generate Face List ---
    faces = []
    verts_per_row = cols if wrap_x else cols + 1
    num_rows = rows if wrap_y else rows + 1
    num_cols = cols if wrap_x else cols + 1

    def get_vertex_index(r, c):
        eff_r = r % rows if wrap_y else r
        eff_c = c % cols if wrap_x else c
        return eff_r * verts_per_row + eff_c

    for r in range(rows):
        for c in range(cols):
            v00, v01 = get_vertex_index(r, c), get_vertex_index(r, c + 1)
            v10, v11 = get_vertex_index(r + 1, c), get_vertex_index(r + 1, c + 1)
            if diagonal_pattern == "v00_v11":
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
            elif diagonal_pattern == "v01_v10":
                faces.append([v00, v10, v01])
                faces.append([v10, v11, v01])
            else:
                raise ValueError(f"Unknown diagonal_pattern: {diagonal_pattern}")

    # --- 2. Create Topology ---
    with factory.update_topology([]) as surgery:
        wb = surgery.workbench
        wb.populate_from_faces(faces)
        surface_id = surgery.commit(f"Populated from {len(faces)} faces.")

    # --- 3. Annotate with Grid Coordinates and a Flat Metric ---
    with factory.update_properties(surface_id) as annotation:
        with annotation.writer() as write_view:
            u_vec, v_vec = np.array([1.0, 0.0]), np.array(v_translation)
            verts_2d = {}

            # This is the brittle part, now isolated. It relies on the current
            # implementation detail that `populate_from_faces` followed by `to_frozen`
            # preserves the order of vertices based on their original integer indices.
            for i, v in enumerate(write_view.vertices):
                r = i // verts_per_row
                c = i % verts_per_row
                v.grid_row = r
                v.grid_col = c
                pos = c * u_vec + r * v_vec
                if perturb_amount > 0.0:
                    np.random.seed(i)
                    pos += (np.random.rand(2) - 0.5) * perturb_amount
                verts_2d[v.id] = pos

            # Calculate and assign edge lengths for this flat metric
            assigned_edges = set()
            wrap_vec_x, wrap_vec_y = cols * u_vec, rows * v_vec
            for he in write_view.half_edges:
                if he.id in assigned_edges:
                    continue

                p1 = verts_2d[he.origin.id]
                p2 = verts_2d[he.next.origin.id]
                delta, min_len_sq = p2 - p1, np.dot(p2 - p1, p2 - p1)

                if wrap_x or wrap_y:
                    for i_offset in ([-1, 0, 1] if wrap_x else [0]):
                        for j_offset in ([-1, 0, 1] if wrap_y else [0]):
                            if i_offset == 0 and j_offset == 0:
                                continue
                            offset = i_offset * wrap_vec_x + j_offset * wrap_vec_y
                            len_sq = np.dot(delta - offset, delta - offset)
                            if len_sq < min_len_sq:
                                min_len_sq = len_sq

                length = np.sqrt(min_len_sq)
                he.length = length
                assigned_edges.add(he.id)
                if he.twin:
                    he.twin.length = length
                    assigned_edges.add(he.twin.id)

    return annotation.commit("Annotated grid coordinates and flat metric.")


# --- PUBLIC-FACING FIXTURES ---


def create_grid(
    factory: SurfaceFactory,
    rows: int = 3,
    cols: int = 3,
    wrap_x: bool = False,
    wrap_y: bool = False,
    equilateral: bool = False,
    perturb_amount: float = 0.0,
) -> SurfaceId:
    """Creates a general grid, optionally wrapping in x and/or y directions."""
    grid_type = "equilateral" if equilateral else "square"
    return _create_triangulated_grid(
        factory,
        rows,
        cols,
        wrap_x,
        wrap_y,
        grid_type=grid_type,
        perturb_amount=perturb_amount,
    )


def create_torus(
    factory: SurfaceFactory,
    rows: int = 3,
    cols: int = 3,
    equilateral: bool = False,
    perturb_amount: float = 0.0,
) -> SurfaceId:
    """Creates a torus (wraps in both x and y)."""
    return create_grid(
        factory,
        rows,
        cols,
        True,
        True,
        equilateral=equilateral,
        perturb_amount=perturb_amount,
    )


def create_annulus(
    factory: SurfaceFactory,
    rows: int = 3,
    cols: int = 3,
    equilateral: bool = False,
    perturb_amount: float = 0.0,
) -> SurfaceId:
    """Creates an annulus (wraps in x but not y)."""
    return create_grid(
        factory,
        rows,
        cols,
        True,
        False,
        equilateral=equilateral,
        perturb_amount=perturb_amount,
    )
