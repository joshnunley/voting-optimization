#!/usr/bin/env python3
"""
Utilities for sampling points along the fixed iso-complexity curves used in the
paper:

    K = a (K0 - 1)^2 (alpha - 1/2 - b/(K0 - 1))^2 + K0

with the published coefficients a=2.35, b=0.29.

The main helper here is `sample_iso_complexity_points`, which:

1. Uses the already-fixed paper curve for a given K0.
2. Finds the valid portion of that curve inside the box
      K in [K_min, K_max], alpha in [alpha_min, alpha_max].
3. Places target locations uniformly in arc length along that valid curve
   segment, treating the segment endpoints as part of the spacing.
4. Returns only the interior sample points.

By default we require K to be an integer, since experiments are run on the
integer NK grid. In that mode, the function projects the continuous interior
arc-length targets onto exact on-curve points with integer K. If there are not
enough distinct integer-K points on the curve segment, it returns as many as
exist.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


PAPER_A = 2.35
PAPER_B = 0.29


@dataclass(frozen=True)
class CurvePoint:
    K0: float
    K: float
    alpha: float
    arc_fraction: float


def complexity_params(K0: float, a: float = PAPER_A, b: float = PAPER_B) -> Tuple[float, float]:
    """
    Return (c, mu) for the paper's fixed complexity curve:

        K(alpha) = c * (alpha - mu)^2 + K0

    For K0=1 the published expression is singular in mu but the curve itself
    degenerates to K(alpha)=1, so we treat mu=0.5 by convention.
    """
    if np.isclose(K0, 1.0):
        return 0.0, 0.5
    c = a * (K0 - 1.0) ** 2
    mu = 0.5 + b / (K0 - 1.0)
    return c, mu


def curve_K(alpha: np.ndarray, K0: float, a: float = PAPER_A, b: float = PAPER_B) -> np.ndarray:
    """Evaluate the paper's iso-complexity curve K(alpha) at fixed K0."""
    c, mu = complexity_params(K0, a=a, b=b)
    if c == 0.0:
        return np.full_like(alpha, float(K0), dtype=float)
    return c * (alpha - mu) ** 2 + float(K0)


def _valid_alpha_interval(
    K0: float,
    K_min: int = 1,
    K_max: int = 20,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    a: float = PAPER_A,
    b: float = PAPER_B,
) -> Optional[Tuple[float, float]]:
    """
    Return the continuous alpha interval where the fixed-K0 curve lies inside
    the bounding box, or None if there is no valid segment.
    """
    c, mu = complexity_params(K0, a=a, b=b)

    if c == 0.0:
        if K_min <= K0 <= K_max:
            return alpha_min, alpha_max
        return None

    if K0 > K_max:
        return None

    max_radius_sq = (K_max - K0) / c
    if max_radius_sq < 0:
        return None

    radius = np.sqrt(max_radius_sq)
    lo = max(alpha_min, mu - radius)
    hi = min(alpha_max, mu + radius)
    if lo > hi:
        return None
    return lo, hi


def _arc_length_grid(
    K0: float,
    alpha_lo: float,
    alpha_hi: float,
    a: float = PAPER_A,
    b: float = PAPER_B,
    resolution: int = 20001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense arc-length parameterization of the valid continuous curve segment.
    Returns alpha_grid and cumulative arc length s(alpha).
    """
    if np.isclose(alpha_lo, alpha_hi):
        alpha_grid = np.array([alpha_lo], dtype=float)
        s_grid = np.array([0.0], dtype=float)
        return alpha_grid, s_grid

    alpha_grid = np.linspace(alpha_lo, alpha_hi, resolution)
    c, mu = complexity_params(K0, a=a, b=b)
    dK_dalpha = np.zeros_like(alpha_grid) if c == 0.0 else 2.0 * c * (alpha_grid - mu)
    ds = np.sqrt(1.0 + dK_dalpha ** 2)
    delta_alpha = np.diff(alpha_grid)
    s_grid = np.concatenate([[0.0], np.cumsum(0.5 * (ds[:-1] + ds[1:]) * delta_alpha)])
    return alpha_grid, s_grid


def _solve_alpha_for_integer_K(
    K: int,
    K0: float,
    a: float = PAPER_A,
    b: float = PAPER_B,
) -> List[float]:
    """
    Solve K = curve(alpha; K0) for alpha, returning all real solutions.
    """
    c, mu = complexity_params(K0, a=a, b=b)

    if c == 0.0:
        if np.isclose(K, K0):
            return [0.0, 1.0]
        return []

    diff = K - K0
    if diff < -1e-12:
        return []

    if np.isclose(diff, 0.0):
        return [mu]

    radius = np.sqrt(diff / c)
    return [mu - radius, mu + radius]


def enumerate_integer_curve_points(
    K0: float,
    K_min: int = 1,
    K_max: int = 20,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    a: float = PAPER_A,
    b: float = PAPER_B,
) -> List[Tuple[float, float]]:
    """
    Enumerate exact on-curve points with integer K and alpha in bounds.

    Returns a sorted list of (alpha, K) points. For K0=1 the valid curve is the
    full line K=1, so this discrete enumeration is not sufficient by itself;
    `sample_iso_complexity_points` handles that case directly.
    """
    pts: List[Tuple[float, float]] = []
    seen = set()
    for K in range(K_min, K_max + 1):
        for alpha in _solve_alpha_for_integer_K(K, K0, a=a, b=b):
            if alpha_min - 1e-12 <= alpha <= alpha_max + 1e-12:
                key = (round(float(alpha), 12), int(K))
                if key not in seen:
                    seen.add(key)
                    pts.append((float(alpha), float(K)))
    pts.sort(key=lambda x: x[0])
    return pts


def sample_iso_complexity_points(
    K0: float,
    num_points: int = 4,
    *,
    integer_K: bool = True,
    K_min: int = 1,
    K_max: int = 20,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    a: float = PAPER_A,
    b: float = PAPER_B,
    resolution: int = 20001,
) -> List[CurvePoint]:
    """
    Sample interior points uniformly in arc length along the fixed K0 curve.

    If integer_K=True, the continuous targets are projected onto exact on-curve
    points with integer K, returning distinct points only. This may yield fewer
    than `num_points` samples near the upper end of the complexity range, where
    the valid curve segment becomes very short on the integer NK grid.
    """
    if num_points <= 0:
        return []

    interval = _valid_alpha_interval(
        K0, K_min=K_min, K_max=K_max, alpha_min=alpha_min, alpha_max=alpha_max, a=a, b=b
    )
    if interval is None:
        return []

    alpha_lo, alpha_hi = interval
    alpha_grid, s_grid = _arc_length_grid(
        K0, alpha_lo, alpha_hi, a=a, b=b, resolution=resolution
    )
    total_length = s_grid[-1]

    if np.isclose(total_length, 0.0):
        alpha = float(alpha_grid[0])
        K = float(curve_K(np.array([alpha]), K0, a=a, b=b)[0])
        if integer_K and not np.isclose(K, round(K)):
            return []
        return [CurvePoint(K0=float(K0), K=float(round(K) if integer_K else K), alpha=alpha, arc_fraction=0.5)]

    target_fracs = np.arange(1, num_points + 1, dtype=float) / (num_points + 1.0)
    target_s = target_fracs * total_length

    if not integer_K:
        alphas = np.interp(target_s, s_grid, alpha_grid)
        Ks = curve_K(alphas, K0, a=a, b=b)
        return [
            CurvePoint(K0=float(K0), K=float(K), alpha=float(alpha), arc_fraction=float(frac))
            for frac, alpha, K in zip(target_fracs, alphas, Ks)
        ]

    # K0=1 is a true horizontal segment K=1, so we can sample directly.
    if np.isclose(K0, 1.0):
        alphas = np.interp(target_s, s_grid, alpha_grid)
        return [
            CurvePoint(K0=float(K0), K=1.0, alpha=float(alpha), arc_fraction=float(frac))
            for frac, alpha in zip(target_fracs, alphas)
        ]

    discrete_pts = enumerate_integer_curve_points(
        K0, K_min=K_min, K_max=K_max, alpha_min=alpha_min, alpha_max=alpha_max, a=a, b=b
    )
    if not discrete_pts:
        return []

    discrete_alpha = np.array([p[0] for p in discrete_pts], dtype=float)
    discrete_K = np.array([p[1] for p in discrete_pts], dtype=float)
    discrete_s = np.interp(discrete_alpha, alpha_grid, s_grid)
    discrete_frac = np.where(total_length > 0, discrete_s / total_length, 0.5)

    interior_mask = (discrete_frac > 1e-9) & (discrete_frac < 1.0 - 1e-9)
    if np.any(interior_mask):
        discrete_alpha = discrete_alpha[interior_mask]
        discrete_K = discrete_K[interior_mask]
        discrete_s = discrete_s[interior_mask]
        discrete_frac = discrete_frac[interior_mask]

    chosen_idx: List[int] = []
    available = list(range(len(discrete_alpha)))
    for s_target in target_s:
        if not available:
            break
        nearest = min(available, key=lambda idx: abs(discrete_s[idx] - s_target))
        chosen_idx.append(nearest)
        available.remove(nearest)

    chosen_idx.sort(key=lambda idx: discrete_alpha[idx])
    return [
        CurvePoint(
            K0=float(K0),
            K=float(discrete_K[idx]),
            alpha=float(discrete_alpha[idx]),
            arc_fraction=float(discrete_frac[idx]),
        )
        for idx in chosen_idx
    ]


def sample_all_K0(
    num_points: int = 4,
    *,
    integer_K: bool = True,
    K0_values: Sequence[int] = tuple(range(1, 21)),
    K_min: int = 1,
    K_max: int = 20,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    a: float = PAPER_A,
    b: float = PAPER_B,
) -> List[List[CurvePoint]]:
    """Sample every integer K0 in the paper's complexity range."""
    return [
        sample_iso_complexity_points(
            float(K0),
            num_points=num_points,
            integer_K=integer_K,
            K_min=K_min,
            K_max=K_max,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            a=a,
            b=b,
        )
        for K0 in K0_values
    ]


if __name__ == "__main__":
    for K0 in range(1, 21):
        pts = sample_iso_complexity_points(K0, num_points=4, integer_K=True)
        printable = [(int(round(p.K)), round(p.alpha, 4), round(p.arc_fraction, 4)) for p in pts]
        print(f"K0={K0:2d}: {printable}")
