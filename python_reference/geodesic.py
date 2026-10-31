# mazedemo/geodesic.py
"""
Intrinsic Geodesic Arc Sampler using Frame-Shifted Möbius Transformations.

Computes exact hyperbolic geodesic arcs in the Poincaré disk D using:
z(t) = m_{z1}^{-1}(t * m_{z1}(z2)) for t in [0, 1]

And straight line segments for Euclidean geometry.
"""

import numpy as np
from typing import List, Union, Tuple, Any
from mazedemo.layout import mobius_transform


def mobius_inverse(w: np.complex128, z_c: np.complex128) -> np.complex128:
    """
    Inverse isometric Möbius transformation:
    m_{z_c}^{-1}(w) = (w + z_c) / (1 + conj(z_c) * w)
    """
    if np.abs(z_c) < 1e-12:
        return w
    num = w + z_c
    den = 1.0 + np.conj(z_c) * w
    if np.abs(den) < 1e-15:
        return w
    return num / den


def compute_hyperbolic_geodesic_arc(
    z1: np.complex128,
    z2: np.complex128,
    num_samples: int = 50,
) -> np.ndarray:
    """
    Computes exact hyperbolic geodesic arc samples in the Poincaré disk between z1 and z2.
    Uses frame-shifted Möbius transformations:
    w2 = m_{z1}(z2)
    z(t) = m_{z1}^{-1}(t * w2),  t in [0, 1]

    Returns:
        Array of complex128 points z(t) of length num_samples.
    """
    z1 = np.complex128(z1)
    z2 = np.complex128(z2)

    if np.abs(z1 - z2) < 1e-12:
        return np.full(num_samples, z1, dtype=np.complex128)

    # Frame shift z1 to origin
    w2 = mobius_transform(z2, z1)

    t_vals = np.linspace(0.0, 1.0, num_samples)
    arc_samples = np.zeros(num_samples, dtype=np.complex128)

    for i, t in enumerate(t_vals):
        w_t = t * w2
        arc_samples[i] = mobius_inverse(w_t, z1)

    return arc_samples


def compute_euclidean_geodesic_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    num_samples: int = 50,
) -> np.ndarray:
    """
    Computes straight line segment samples in Euclidean space between p1 and p2.

    Returns:
        Array of N x 2 float64 points.
    """
    p1 = np.asarray(p1, dtype=np.float64)[:2]
    p2 = np.asarray(p2, dtype=np.float64)[:2]

    t_vals = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]
    return (1.0 - t_vals) * p1 + t_vals * p2
