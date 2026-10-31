from typing import List, Tuple

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.viewer import Viewer
from mazedemo.ids import SurfaceId


def get_surface_fixture_data(viewer: Viewer) -> Tuple[List[List[int]], List[float]]:
    """Extracts topology (faces) and geometry (lengths) from a surface viewer."""
    vertex_map = {v.id: i for i, v in enumerate(viewer.vertices)}

    faces_data = []
    for face in viewer.faces:
        face_verts = [
            vertex_map[he.origin.id] for he in viewer.get_face_half_edges(face.id)
        ]
        faces_data.append(face_verts)

    # To be deterministic, sort half-edges.
    sorted_hes = sorted(
        viewer.half_edges,
        key=lambda he: (vertex_map[he.origin.id], vertex_map[he.next.origin.id]),
    )

    lengths_data = [he.get_property("length") for he in sorted_hes]

    return faces_data, lengths_data


def save_surface_to_fixture_file(viewer: Viewer, file_path: str):
    """Saves surface data as a Python fixture file."""
    faces, lengths = get_surface_fixture_data(viewer)

    with open(file_path, "w") as f:
        f.write("FACES = [\n")
        for face in faces:
            f.write(f"    {face},\n")
        f.write("]\n\n")

        f.write("LENGTHS = [\n")
        for length in lengths:
            f.write(f"    {length},\n")
        f.write("]\n")


def load_surface_from_fixture(
    factory: SurfaceFactory, faces: List[List[int]], lengths: List[float]
) -> SurfaceId:
    """Creates a surface from fixture data (faces and lengths)."""
    with factory.update_topology() as surgery:
        surgery.workbench.populate_from_faces(faces)
        topo_surface_id = surgery.commit("Create from fixture topology")

    with factory.update_properties(topo_surface_id) as annotation:
        with annotation.writer() as view:
            # Re-create the same canonical sort order used by the generation script.
            vertex_map = {v.id: i for i, v in enumerate(view.vertices)}
            sorted_hes = sorted(
                view.half_edges,
                key=lambda he: (
                    vertex_map[he.origin.id],
                    vertex_map[he.next.origin.id],
                ),
            )

            if len(sorted_hes) != len(lengths):
                raise ValueError(
                    "Mismatch between number of half-edges in fixture topology and number of lengths."
                )

            for he, length in zip(sorted_hes, lengths):
                he.length = length

            metric_surface_id = annotation.commit("Apply fixture metric")

    return metric_surface_id
