import sys
import os
import copy
import time
import signal
from contextlib import contextmanager

import networkx as nx
import numpy as np
from absl import app
from absl import flags
from absl import logging
from matplotlib import pyplot as plt

from mazedemo import (
    mazeutil,
    surface_factory,
    discrete_conformal_map,
    fundamental_domain,
    euclidean,
    hyperbolic,
    vis,
)
from mazedemo.serialize import save_surface_to_fixture_file

FLAGS = flags.FLAGS
flags.DEFINE_integer("rows", 3, "Number of rows in the maze grid.")
flags.DEFINE_integer("cols", 3, "Number of columns in the maze grid.")
flags.DEFINE_integer(
    "genus", 1, "Genus of the maze, defined as the number of interior holes."
)
flags.DEFINE_integer("timeout", 60, "Timeout in seconds for long-running operations.")
flags.DEFINE_string(
    "output_dir", "/tmp/bad_mazes", "Directory to save problematic surfaces."
)
flags.DEFINE_boolean("visualize", False, "Whether to plot the fundamental domain.")


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _generate_maze_graph(rows, cols, num_boundary_holes, num_interior_holes):
    """Generates a maze graph with a specified number of holes."""
    g = nx.grid_graph(dim=(rows + 1, cols + 1))
    g_dual = nx.Graph()
    g_dual.add_edges_from(
        [
            ((i, j), (i + 1, j), {"dual": ((i + 1, j), (i + 1, j + 1))})
            for i in range(cols - 1)
            for j in range(rows)
        ]
    )
    g_dual.add_edges_from(
        [
            ((i, j), (i, j + 1), {"dual": ((i, j + 1), (i + 1, j + 1))})
            for i in range(cols)
            for j in range(rows - 1)
        ]
    )
    st = mazeutil.ust(g_dual)
    g.remove_edges_from(nx.get_edge_attributes(st, "dual").values())

    exterior_edges = [
        e
        for e in g.edges
        if (
            (e[0][0] % (cols + 1) == 0 and e[1][0] % (cols + 1) == 0)
            or (e[0][1] % (rows + 1) == 0 and e[1][1] % (rows + 1) == 0)
        )
    ]
    ext = frozenset(exterior_edges)
    interior_edges = [e for e in g.edges if e not in ext]

    for attempt in range(10):
        g_with_holes = copy.deepcopy(g)
        if num_boundary_holes > 0:
            g_with_holes.remove_edges_from(
                [
                    e.o
                    for e in np.random.choice(
                        [mazeutil.Wrap(e) for e in exterior_edges],
                        num_boundary_holes,
                        replace=False,
                    )
                ]
            )
        if num_interior_holes > 0:
            g_with_holes.remove_edges_from(
                [
                    e.o
                    for e in np.random.choice(
                        [mazeutil.Wrap(e) for e in interior_edges],
                        num_interior_holes,
                        replace=False,
                    )
                ]
            )

        isolated_nodes = list(nx.isolates(g_with_holes))
        if isolated_nodes:
            logging.warning(
                f"#{attempt}: isolated vertex - retrying ({isolated_nodes})"
            )
        else:
            return g_with_holes

    raise RuntimeError("Failed to generate a maze without isolated vertices.")


def run_maze(factory, rows, cols, genus, timeout, output_dir, visualize):
    """Generates and processes a maze with the specified geometry."""
    num_boundary_holes = 1
    num_interior_holes = genus

    if genus <= 1:
        geometry_str = "euclidean"
        geometry = euclidean.EuclideanGeometry()
    else:  # genus > 1
        geometry_str = "hyperbolic"
        geometry = hyperbolic.HyperbolicGeometry()

    logging.info(
        f"Generating genus {genus} maze ({rows}x{cols}) with {geometry_str} geometry..."
    )
    g = _generate_maze_graph(rows, cols, num_boundary_holes, num_interior_holes)

    t = mazeutil.triangulate_complement(factory, g)
    surface_ids = [
        d["surface_id"] for n, d in t.nodes(data=True) if d.get("type") == "geometric"
    ]
    surface_with_lengths_id = mazeutil.annotate_lengths_from_positions(
        factory, surface_ids[0]
    )

    surface_to_save = surface_with_lengths_id
    operation_name = "discrete_conformal_map"
    try:
        with time_limit(timeout):
            logging.info("Flattening triangulation...")
            flat_surface_id = discrete_conformal_map.discrete_conformal_map(
                factory, surface_with_lengths_id, geometry=geometry_str
            )

        surface_to_save = flat_surface_id
        operation_name = "compute_fundamental_domain"
        with time_limit(timeout):
            logging.info("Generating fundamental domain...")
            domain_artifacts = fundamental_domain.compute_fundamental_domain(
                factory, flat_surface_id, geometry
            )

        if visualize:
            logging.info("Visualizing fundamental domain...")
            (
                dual_graph,
                centered_embeddings,
                isometries,
                spanning_tree,
                holonomy_classes,
                placed_triangles,
                holonomy_transforms,
                generators,
            ) = domain_artifacts
            vis.plot_fundamental_domain(
                placed_triangles,
                spanning_tree,
                holonomy_transforms,
                geometry,
                title=f"{geometry_str.capitalize()} Maze Domain",
            )
            plt.show()

    except (TimeoutException, Exception) as e:
        logging.error(f"Operation '{operation_name}' failed: {e}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"genus{genus}_{rows}x{cols}_{operation_name}_{timestamp}.py"
        filepath = os.path.join(output_dir, filename)

        logging.info(f"Saving problematic surface to {filepath}")
        viewer = factory.create_viewer(surface_to_save)
        save_surface_to_fixture_file(viewer, filepath)
        return

    logging.info("Successfully processed maze.")


def main(argv):
    del argv  # Unused.
    factory = surface_factory.SurfaceFactory()
    run_maze(
        factory,
        FLAGS.rows,
        FLAGS.cols,
        FLAGS.genus,
        FLAGS.timeout,
        FLAGS.output_dir,
        FLAGS.visualize,
    )


if __name__ == "__main__":
    app.run(main)
