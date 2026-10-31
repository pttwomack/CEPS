import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from absl.testing import absltest
from absl import logging
import numpy as np
import time

from mazedemo.grid_fixtures import (
    create_torus,
    create_annulus,
)
from mazedemo.test_fixtures import create_flattened_genus_two_surface
from mazedemo.discrete_conformal_map import discrete_conformal_map
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import SurfaceId
from mazedemo.subdivide import subdivide
from mazedemo.hyperbolic import HyperbolicGeometry


class BenchmarkLargeGrids(absltest.TestCase):
    def setUp(self):
        self.factory = SurfaceFactory()

    def test_large_torus_uniformization(self):
        """Benchmarks flattening perturbed tori of increasing size."""
        self.sizes = [int(s) for s in os.environ.get("SIZES", "10,20,50").split(",") if s]
        for size in self.sizes:
            with self.subTest(size=size):
                logging.info(f"Benchmarking torus of size {size}x{size}")
                start_time = time.time()
                torus_id = create_torus(
                    self.factory, rows=size, cols=size, perturb_amount=0.1
                )
                target_curvature = np.zeros(
                    len(self.factory.create_viewer(torus_id).vertices)
                )
                _id, final_gradient = discrete_conformal_map(
                    self.factory, torus_id, target_curvature, return_gradient=True
                )
                duration = time.time() - start_time
                logging.info(f"Torus {size}x{size} took {duration:.2f} seconds.")
                self.assertTrue(
                    np.allclose(final_gradient, 0.0, atol=1e-5),
                    f"Optimization failed for size {size}. Gradient norm: {np.linalg.norm(final_gradient):.4e}",
                )

    def test_large_annulus_uniformization(self):
        """Benchmarks flattening perturbed annuli of increasing size."""
        sizes = [10, 20, 50, 100]
        self.sizes = [int(s) for s in os.environ.get("SIZES", "10,20,50").split(",") if s]
        for size in self.sizes:
            with self.subTest(size=size):
                logging.info(f"Benchmarking annulus of size {size}x{size}")
                start_time = time.time()
                annulus_id = create_annulus(
                    self.factory, rows=size, cols=size, perturb_amount=1.0
                )
                target_curvature = np.zeros(
                    len(self.factory.create_viewer(annulus_id).vertices)
                )
                _id, final_gradient = discrete_conformal_map(
                    self.factory, annulus_id, target_curvature, return_gradient=True
                )
                duration = time.time() - start_time
                logging.info(f"Annulus {size}x{size} took {duration:.2f} seconds.")
                self.assertTrue(
                    np.allclose(final_gradient, 0.0, atol=1e-5),
                    f"Optimization failed for size {size}. Gradient norm: {np.linalg.norm(final_gradient):.4e}",
                )


class BenchmarkGenus2(absltest.TestCase):
    def setUp(self):
        self.factory = SurfaceFactory()
        # SIZES env var can be e.g. "0,1,2,3" for subdivision levels
        self.sizes = [int(s) for s in os.environ.get("SIZES", "0,1,2").split(",") if s]
        self.geometry = HyperbolicGeometry()

    def test_genus2_subdivision_uniformization(self):
        """Benchmarks flattening a genus-2 surface at various subdivision levels."""
        logging.info("Creating base genus-2 surface...")
        surface_id = create_flattened_genus_two_surface(self.factory)

        for i in range(max(self.sizes) + 1):
            if i in self.sizes:
                with self.subTest(subdivision_level=i):
                    viewer = self.factory.create_viewer(surface_id)
                    num_faces = len(viewer.faces)
                    logging.info(
                        f"Benchmarking genus-2 surface, level {i} ({num_faces} faces)..."
                    )
                    start_time = time.time()
                    target_curvature = np.zeros(len(viewer.vertices))
                    _id, final_gradient = discrete_conformal_map(
                        self.factory,
                        surface_id,
                        target_curvature,
                        return_gradient=True,
                        geometry="hyperbolic",
                    )
                    duration = time.time() - start_time
                    logging.info(f"Genus-2 level {i} took {duration:.2f} seconds.")
                    self.assertTrue(
                        np.allclose(final_gradient, 0.0, atol=1e-5),
                        f"Optimization failed for level {i}. Gradient norm: {np.linalg.norm(final_gradient):.4e}",
                    )

            if i < max(self.sizes):
                logging.vlog(1, f"Subdividing to level {i+1}...")
                surface_id = subdivide(self.factory, surface_id, self.geometry)


if __name__ == "__main__":
    absltest.main()
