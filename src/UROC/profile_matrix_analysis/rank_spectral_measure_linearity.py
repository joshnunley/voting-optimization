#!/usr/bin/env python3
"""
Rank primitive spectral measures by how linearly they track K0.

Uses the 30-sample raw profile-matrix dataset and evaluates K0-averaged curves
for:
  - spectral entropies
  - effective ranks / participation ratios
  - primitive spectral-shape measures from raw, candidate-centered, and
    voter-centered spectra

Outputs a CSV of measures sorted by linear-fit R^2, plus a small plot of the
top few most linear measures.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


def parse_args():
    p = argparse.ArgumentParser(description="Rank spectral measures by linearity vs K0")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/raw_profile_matrix_analysis.npz",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/spectral_linearity",
    )
    return p.parse_args()


def entropy_from_weights(w):
    w = np.asarray(w, dtype=float)
    w = w[w > 1e-15]
    if len(w) == 0:
        return 0.0
    p = w / np.sum(w)
    return float(-np.sum(p * np.log(p)))


def primitive_shape_measures(weights, prefix):
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if total <= 0:
        p = np.zeros_like(w)
    else:
        p = w / total
    p = np.pad(p, (0, max(0, 4 - len(p))), constant_values=0.0)
    p1, p2, p3, p4 = p[:4]
    return {
        f"{prefix}_share1": p1,
        f"{prefix}_share2": p2,
        f"{prefix}_share3": p3,
        f"{prefix}_top2": p1 + p2,
        f"{prefix}_tail1": 1.0 - p1,
        f"{prefix}_tail2": 1.0 - (p1 + p2),
        f"{prefix}_gap12": p1 - p2,
        f"{prefix}_gap23": p2 - p3,
        f"{prefix}_ratio21": p2 / max(p1, 1e-12),
        f"{prefix}_ratio31": p3 / max(p1, 1e-12),
        f"{prefix}_ratio32": p3 / max(p2, 1e-12),
        f"{prefix}_effective_rank": float(np.exp(entropy_from_weights(p))),
        f"{prefix}_participation": float(1.0 / max(np.sum(p ** 2), 1e-12)),
        f"{prefix}_entropy": float(entropy_from_weights(p)),
    }


def all_spectral_measures(U):
    out = {}

    s = np.linalg.svd(U, compute_uv=False)
    out.update(primitive_shape_measures(np.square(s), "raw"))

    Xc = U - U.mean(axis=0, keepdims=True)
    cand = np.linalg.eigvalsh(Xc.T @ Xc)
    cand = np.clip(np.sort(cand)[::-1], 0.0, None)
    out.update(primitive_shape_measures(cand, "cand"))

    Xv = U - U.mean(axis=1, keepdims=True)
    voter = np.linalg.eigvalsh(Xv.T @ Xv)
    voter = np.clip(np.sort(voter)[::-1], 0.0, None)
    out.update(primitive_shape_measures(voter, "voter"))
    return out


def linear_fit_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coef = np.polyfit(x, y, 1)
    yhat = coef[0] * x + coef[1]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rho, _ = spearmanr(x, y)
    return coef[0], coef[1], float(r2), float(rho), yhat


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    matrices = d["matrices"].astype(float)
    k0 = d["K0_values"].astype(int)

    rows = []
    names = None
    for i in range(matrices.shape[0]):
        for j in range(matrices.shape[1]):
            feats = all_spectral_measures(matrices[i, j])
            if names is None:
                names = list(feats.keys())
            rows.append([feats[n] for n in names])
    sample_matrix = np.array(rows, dtype=float)
    mean_by_k0 = sample_matrix.reshape(len(k0), matrices.shape[1], len(names)).mean(axis=1)

    result_rows = []
    top_plot = []
    for idx, name in enumerate(names):
        y = mean_by_k0[:, idx]
        slope, intercept, r2, rho, yhat = linear_fit_stats(k0, y)
        result_rows.append({
            "measure": name,
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "abs_spearman": float(abs(rho)),
            "spearman": float(rho),
        })
        top_plot.append((name, normalize_curve(y), normalize_curve(yhat), r2))

    result_rows.sort(key=lambda r: (-r["r2"], -r["abs_spearman"], r["measure"]))
    top_names = [r["measure"] for r in result_rows[:8]]
    write_csv(
        outdir / "spectral_measure_linearity.csv",
        result_rows,
        ["measure", "slope", "intercept", "r2", "abs_spearman", "spearman"],
    )

    lookup = {name: (curve, fit, r2) for name, curve, fit, r2 in top_plot}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]
    for color, name in zip(palette, top_names):
        curve, fit, r2 = lookup[name]
        ax.plot(k0, curve, color=color, linewidth=2.0, label=f"{name} (R²={r2:.3f})")
    ax.set_xlabel("K0")
    ax.set_ylabel("normalized value")
    ax.set_title("Most linear spectral measures vs K0")
    ax.set_xticks(k0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "top_linear_spectral_measures.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Top 10 most linear spectral measures:")
    for row in result_rows[:10]:
        print(f"  {row['measure']}: R2={row['r2']:.3f}, spearman={row['spearman']:.3f}")
    print(f"saved {outdir / 'spectral_measure_linearity.csv'}")


if __name__ == "__main__":
    main()
