#!/usr/bin/env python3
"""
Search for regime-specialized measures using only primitive spectral-shape
descriptors. No bell transforms, no inversions, no composite products.

Spectral families used:
1. raw singular-value spectrum of U
2. candidate-centered spectrum of (U - colmean)^T (U - colmean)
3. voter-centered nonzero spectrum of (U - rowmean)^T (U - rowmean)

For each spectrum, we derive primitive shape measures such as:
  - top share
  - second share
  - tail share
  - spectral gaps
  - simple ratios
  - effective rank
  - participation ratio

We then search for 3-measure and 4-measure sets whose normalized K0-mean
curves specialize to the target regimes.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES_3 = [
    ("early", (3, 8)),
    ("mid", (9, 14)),
    ("high", (15, 20)),
]

REGIMES_4 = [
    ("trivial", (1, 2)),
    ("early", (3, 8)),
    ("mid", (9, 14)),
    ("high", (15, 20)),
]


def parse_args():
    p = argparse.ArgumentParser(description="Search spectral-only regime measures")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/raw_profile_matrix_analysis.npz",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan_30samples/spectral_shape_search",
    )
    p.add_argument("--top_n", type=int, default=50)
    return p.parse_args()


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
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
    measures = {
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
        f"{prefix}_effective_rank": float(np.exp(entropy(p))),
        f"{prefix}_participation": float(1.0 / max(np.sum(p ** 2), 1e-12)),
    }
    return measures


def compute_spectral_measure_row(U):
    out = {}

    s = np.linalg.svd(U, compute_uv=False)
    out.update(primitive_shape_measures(np.square(s), "raw"))

    Xc = U - U.mean(axis=0, keepdims=True)
    cand = np.linalg.eigvalsh(Xc.T @ Xc)
    cand = np.clip(np.sort(cand)[::-1], 0.0, None)
    out.update(primitive_shape_measures(cand, "cand"))

    Xv = U - U.mean(axis=1, keepdims=True)
    voter_small = np.linalg.eigvalsh(Xv.T @ Xv)
    voter_small = np.clip(np.sort(voter_small)[::-1], 0.0, None)
    out.update(primitive_shape_measures(voter_small, "voter"))

    return out


def masks_from_regimes(k0_values, regimes):
    out = []
    for name, (lo, hi) in regimes:
        out.append((name, (k0_values >= lo) & (k0_values <= hi)))
    return out


def specialization_score(curve, target_mask, other_masks):
    return float(np.mean(curve[target_mask]) - np.mean(np.concatenate([curve[m] for m in other_masks])))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_measure_set(k0_values, curves, picked_names, regimes, outpath, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#666666", "#377eb8", "#4daf4a", "#e41a1c"]
    for color, (reg_name, _), curve_name in zip(palette, regimes, picked_names):
        ax.plot(k0_values, curves[curve_name], linewidth=2.2, color=color, label=f"{reg_name}: {curve_name}")
    for color, (_, (lo, hi)) in zip(palette, regimes):
        ax.axvspan(lo, hi, color=color, alpha=0.08)
    ax.set_xlabel("K0")
    ax.set_ylabel("normalized spectral shape measure")
    ax.set_title(title)
    ax.set_xticks(k0_values)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def search_best_set(curves, k0_values, regimes, top_n):
    masks = masks_from_regimes(k0_values, regimes)
    curve_names = list(curves.keys())
    n = len(regimes)
    rows = []
    best = None
    best_total = -1e18

    for combo in itertools.permutations(curve_names, n):
        if len(set(combo)) < n:
            continue
        scores = []
        row = {}
        for i, (reg_name, mask) in enumerate(masks):
            other_masks = [m for j, (_, m) in enumerate(masks) if j != i]
            sc = specialization_score(curves[combo[i]], mask, other_masks)
            row[f"{reg_name}_measure"] = combo[i]
            row[f"{reg_name}_score"] = float(sc)
            scores.append(sc)
        total = float(np.sum(scores))
        row["total_score"] = total
        rows.append(row)
        if total > best_total:
            best_total = total
            best = row

    fieldnames = []
    for reg_name, _ in regimes:
        fieldnames.extend([f"{reg_name}_measure", f"{reg_name}_score"])
    fieldnames.append("total_score")
    rows.sort(key=lambda r: (-r["total_score"],) + tuple(r[f"{reg_name}_measure"] for reg_name, _ in regimes))
    return best, rows[:top_n], fieldnames


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    matrices = d["matrices"].astype(float)
    k0_values = d["K0_values"].astype(int)

    # Build per-sample spectral features
    sample_rows = []
    feature_names = None
    for i, K0 in enumerate(k0_values):
        for j in range(matrices.shape[1]):
            feats = compute_spectral_measure_row(matrices[i, j])
            if feature_names is None:
                feature_names = list(feats.keys())
            sample_rows.append([feats[name] for name in feature_names])
    sample_matrix = np.array(sample_rows, dtype=float)

    # Mean curve by K0 for each primitive spectral feature
    features_by_k0 = sample_matrix.reshape(len(k0_values), matrices.shape[1], len(feature_names)).mean(axis=1)
    curves = {name: normalize_curve(features_by_k0[:, idx]) for idx, name in enumerate(feature_names)}

    best3, top3, fields3 = search_best_set(curves, k0_values, REGIMES_3, args.top_n)
    best4, top4, fields4 = search_best_set(curves, k0_values, REGIMES_4, args.top_n)

    write_csv(outdir / "top_spectral_sets_3.csv", top3, fields3)
    write_csv(outdir / "top_spectral_sets_4.csv", top4, fields4)

    picked3 = [best3[f"{name}_measure"] for name, _ in REGIMES_3]
    picked4 = [best4[f"{name}_measure"] for name, _ in REGIMES_4]
    plot_measure_set(
        k0_values, curves, picked3, REGIMES_3,
        outdir / "best_spectral_set_3.pdf",
        title="Best 3-regime spectral-shape set",
    )
    plot_measure_set(
        k0_values, curves, picked4, REGIMES_4,
        outdir / "best_spectral_set_4.pdf",
        title="Best 4-regime spectral-shape set",
    )

    np.savez_compressed(
        outdir / "spectral_shape_search.npz",
        k0_values=k0_values,
        feature_names=np.array(feature_names, dtype=object),
        feature_curves=np.vstack([curves[name] for name in feature_names]),
        best3=np.array([best3[f"{name}_measure"] for name, _ in REGIMES_3], dtype=object),
        best4=np.array([best4[f"{name}_measure"] for name, _ in REGIMES_4], dtype=object),
    )

    print("Best 3-regime spectral set:")
    for name, _ in REGIMES_3:
        print(f"  {name}: {best3[f'{name}_measure']}  score={best3[f'{name}_score']:.3f}")
    print(f"  total={best3['total_score']:.3f}")

    print("Best 4-regime spectral set:")
    for name, _ in REGIMES_4:
        print(f"  {name}: {best4[f'{name}_measure']}  score={best4[f'{name}_score']:.3f}")
    print(f"  total={best4['total_score']:.3f}")
    print(f"saved {outdir / 'spectral_shape_search.npz'}")


if __name__ == "__main__":
    main()
