# mazedemo/run_vis.py
"""A script to generate and save visualizations of polyhedral surfaces."""

"""A script to generate and save visualizations of polyhedral surfaces."""
import sys
import os
import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mazedemo.surface_factory import SurfaceFactory
from mazedemo.test_fixtures import (
    create_dodecahedron,
    create_octahedron,
    create_torus_fundamental_domain,
)
from mazedemo.vis import plot_dcel, plot_fundamental_domain

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "figure",
    "dodecahedron",
    ["dodecahedron", "octahedron", "torus_domain"],
    "Which item to visualize.",
)
flags.DEFINE_string(
    "output_dir", "_visualizations", "The directory to save the output image."
)


def main(argv):
    """Generates the selected item and plots it."""
    factory = SurfaceFactory()

    if FLAGS.figure == "torus_domain":
        print("--- Visualizing Torus Fundamental Domain ---")

        torus_id, domain_artifacts = create_torus_fundamental_domain(factory)
        (
            dual_graph,
            centered_embeddings,
            isometries,
            spanning_tree,
            holonomy_classes,
            placements,
            holonomy_transforms,
            generators,
        ) = domain_artifacts

        view = factory.create_viewer(torus_id)
        print(f"Plotting fundamental domain for {view}")
        plot_fundamental_domain(
            placements,
            spanning_tree,
            holonomy_transforms,
            title="Torus_Fundamental_Domain",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(FLAGS.output_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved visualization to {filename}")
        return

    if FLAGS.figure == "dodecahedron":
        print("--- Visualizing Dodecahedron Surface ---")
        surface_id = create_dodecahedron(factory)
        title = "Dodecahedron_Stereographic_Projection"
    elif FLAGS.figure == "octahedron":
        print("--- Visualizing Octahedron Surface ---")
        surface_id = create_octahedron(factory)
        title = "Octahedron_Stereographic_Projection"
    else:
        raise ValueError(f"Unknown figure: {FLAGS.figure}")

    viewer = factory.create_viewer(surface_id)
    plot_dcel(viewer, title=title, explode_face_id=0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(FLAGS.output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    app.run(main)
