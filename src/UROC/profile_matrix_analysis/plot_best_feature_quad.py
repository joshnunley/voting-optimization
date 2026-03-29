#!/usr/bin/env python3
"""
Plot the current best four-curve regime picture from the raw profile-matrix
analysis:

1. trivial early simplicity      : cand_eig_1
2. genuine early regime          : bell(sv_energy_1)
3. middle / Borda-like regime    : bell(sv_energy_1) * inv(cand_eig_1)
4. high-complexity fragmentation : inv(cand_eig_1)
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot the best four-feature regime picture")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/raw_profile_matrix_analysis.npz",
    )
    p.add_argument(
        "--output",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/regime_feature_triples/best_feature_quad.pdf",
    )
    return p.parse_args()


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def main():
    args = parse_args()
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    feature_names = list(d["feature_names"])
    features_by_k0 = d["features_by_k0"].astype(float)
    k0 = d["K0_values"].astype(int)
    mean_by_k0 = features_by_k0.mean(axis=1)

    cand_eig_1 = normalize_curve(mean_by_k0[:, feature_names.index("cand_eig_1")])
    sv_energy_1 = normalize_curve(mean_by_k0[:, feature_names.index("sv_energy_1")])

    bell_sv1 = 4.0 * sv_energy_1 * (1.0 - sv_energy_1)
    inv_cand = 1.0 - cand_eig_1
    middle = normalize_curve(bell_sv1 * inv_cand)

    curves = [
        ("trivial early: cand_eig_1", cand_eig_1, "#666666"),
        ("early: bell(sv_energy_1)", bell_sv1, "#377eb8"),
        ("middle: bell(sv_energy_1) * inv(cand_eig_1)", middle, "#4daf4a"),
        ("high: inv(cand_eig_1)", inv_cand, "#e41a1c"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, curve, color in curves:
        ax.plot(k0, curve, linewidth=2.3, color=color, label=label)

    ax.axvspan(1, 2, color="#666666", alpha=0.08)
    ax.axvspan(3, 8, color="#377eb8", alpha=0.08)
    ax.axvspan(8, 15, color="#4daf4a", alpha=0.08)
    ax.axvspan(15, 20, color="#e41a1c", alpha=0.08)

    ax.set_xlabel("K0")
    ax.set_ylabel("normalized score")
    ax.set_title("Best feature quad from raw profile-matrix diagnostics")
    ax.set_xticks(k0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {outpath}")


if __name__ == "__main__":
    main()
