#!/usr/bin/env python3
"""
Search for triples of normalized feature curves whose regime-specialized
components are maximized in the low / mid / high complexity bands.

This works on the K0-aggregated feature curves derived from the raw profile
matrix analysis output. It is designed to answer the question:

    Is there a small triple of profile-level quantities such that one is most
    active in the low regime, one in the middle, and one in the high regime?

We allow simple transformed variants of primitive features so the "middle"
component can be truly middle-peaked rather than forced to be monotone.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOW_RANGE = (3, 8)
MID_RANGE = (9, 14)
HIGH_RANGE = (15, 20)


def parse_args():
    p = argparse.ArgumentParser(description="Search for low/mid/high regime feature triples")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/raw_profile_matrix_analysis.npz",
        help="Input feature dataset from analyze_raw_profile_matrices.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/regime_feature_triples",
        help="Directory for outputs",
    )
    p.add_argument(
        "--max_feature_pool",
        type=int,
        default=15,
        help="Restrict the primitive feature pool to the top-N candidate features from the clustering pass",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top triples to write out",
    )
    return p.parse_args()


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def regime_masks(k0_values):
    k0 = np.asarray(k0_values, dtype=int)
    low = (k0 >= LOW_RANGE[0]) & (k0 <= LOW_RANGE[1])
    mid = (k0 >= MID_RANGE[0]) & (k0 <= MID_RANGE[1])
    high = (k0 >= HIGH_RANGE[0]) & (k0 <= HIGH_RANGE[1])
    return low, mid, high


def specialization_score(curve, target_mask, other_masks):
    target = float(np.mean(curve[target_mask]))
    other = float(np.mean(np.concatenate([curve[m] for m in other_masks])))
    return target - other


def sharpness_score(curve, target_mask):
    inside = float(np.mean(curve[target_mask]))
    outside = float(np.mean(curve[~target_mask]))
    return inside - outside


def build_candidate_curves(feature_names, mean_by_k0):
    """
    Build a dictionary of candidate regime curves from primitive K0-averaged
    feature curves.

    Variants:
    - raw normalized feature
    - inverted normalized feature
    - bell transform 4x(1-x), which naturally peaks in the middle
    """
    curves = {}
    for name, y in zip(feature_names, mean_by_k0.T):
        x = normalize_curve(y)
        curves[f"{name}"] = x
        curves[f"inv({name})"] = 1.0 - x
        curves[f"bell({name})"] = 4.0 * x * (1.0 - x)
    return curves


def read_top_feature_pool(input_npz: Path, max_feature_pool: int):
    top_csv = input_npz.parent / "clustering" / "top_discriminative_features.csv"
    if top_csv.exists():
        rows = list(csv.DictReader(top_csv.open()))
        feats = [r["feature"] for r in rows[:max_feature_pool]]
        if feats:
            return feats
    return None


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_best_triple(k0_values, curves, triple_names, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "low": "#377eb8",
        "mid": "#4daf4a",
        "high": "#e41a1c",
    }
    for regime, name in zip(["low", "mid", "high"], triple_names):
        ax.plot(k0_values, curves[name], label=f"{regime}: {name}", color=colors[regime], linewidth=2.2)
    ax.axvspan(LOW_RANGE[0], LOW_RANGE[1], color=colors["low"], alpha=0.08)
    ax.axvspan(MID_RANGE[0], MID_RANGE[1], color=colors["mid"], alpha=0.08)
    ax.axvspan(HIGH_RANGE[0], HIGH_RANGE[1], color=colors["high"], alpha=0.08)
    ax.set_xlabel("K0")
    ax.set_ylabel("normalized regime score")
    ax.set_title("Best low/mid/high feature triple")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input_npz, allow_pickle=True)
    feature_names = list(d["feature_names"])
    features_by_k0 = d["features_by_k0"].astype(float)
    k0_values = d["K0_values"].astype(int)

    mean_by_k0 = features_by_k0.mean(axis=1)

    primitive_pool = read_top_feature_pool(Path(args.input_npz), args.max_feature_pool)
    if primitive_pool is None:
        primitive_pool = feature_names[:args.max_feature_pool]
    keep_idx = [feature_names.index(name) for name in primitive_pool]
    feature_names = [feature_names[i] for i in keep_idx]
    mean_by_k0 = mean_by_k0[:, keep_idx]

    curves = build_candidate_curves(feature_names, mean_by_k0)
    curve_names = list(curves.keys())

    low_mask, mid_mask, high_mask = regime_masks(k0_values)
    score_rows = []
    best = None
    best_total = -1e18

    low_candidates = [name for name in curve_names if not name.startswith("bell(")]
    high_candidates = [name for name in curve_names if not name.startswith("bell(")]
    mid_candidates = curve_names

    for low_name in low_candidates:
        low_curve = curves[low_name]
        low_score = specialization_score(low_curve, low_mask, [mid_mask, high_mask])
        for mid_name in mid_candidates:
            if mid_name == low_name:
                continue
            mid_curve = curves[mid_name]
            mid_score = specialization_score(mid_curve, mid_mask, [low_mask, high_mask])
            for high_name in high_candidates:
                if high_name == low_name or high_name == mid_name:
                    continue
                high_curve = curves[high_name]
                high_score = specialization_score(high_curve, high_mask, [low_mask, mid_mask])

                monotonic_ok = (
                    np.mean(low_curve[low_mask]) > np.mean(low_curve[high_mask]) and
                    np.mean(high_curve[high_mask]) > np.mean(high_curve[low_mask])
                )
                if not monotonic_ok:
                    continue

                total = low_score + mid_score + high_score
                row = {
                    "low_curve": low_name,
                    "mid_curve": mid_name,
                    "high_curve": high_name,
                    "low_score": float(low_score),
                    "mid_score": float(mid_score),
                    "high_score": float(high_score),
                    "total_score": float(total),
                    "mid_sharpness": float(sharpness_score(mid_curve, mid_mask)),
                }
                score_rows.append(row)
                if total > best_total:
                    best_total = total
                    best = row

    score_rows.sort(key=lambda r: (-r["total_score"], -r["mid_sharpness"], r["low_curve"], r["mid_curve"], r["high_curve"]))
    write_csv(
        outdir / "top_feature_triples.csv",
        score_rows[:args.top_n],
        ["low_curve", "mid_curve", "high_curve", "low_score", "mid_score", "high_score", "mid_sharpness", "total_score"],
    )

    np.savez_compressed(
        outdir / "feature_triple_search.npz",
        k0_values=k0_values,
        curve_names=np.array(curve_names, dtype=object),
        curve_matrix=np.vstack([curves[name] for name in curve_names]),
        best_low=np.array(best["low_curve"], dtype=object),
        best_mid=np.array(best["mid_curve"], dtype=object),
        best_high=np.array(best["high_curve"], dtype=object),
        best_total=np.array(best["total_score"]),
    )

    plot_best_triple(
        k0_values,
        curves,
        [best["low_curve"], best["mid_curve"], best["high_curve"]],
        outdir / "best_feature_triple.pdf",
    )

    print("Best regime triple:")
    print(f"  low : {best['low_curve']}  (score={best['low_score']:.3f})")
    print(f"  mid : {best['mid_curve']}  (score={best['mid_score']:.3f})")
    print(f"  high: {best['high_curve']} (score={best['high_score']:.3f})")
    print(f"  total={best['total_score']:.3f}")
    print("Top 10 triples:")
    for row in score_rows[:10]:
        print(
            f"  low={row['low_curve']} | mid={row['mid_curve']} | high={row['high_curve']} "
            f":: total={row['total_score']:.3f}"
        )
    print(f"saved {outdir / 'feature_triple_search.npz'}")


if __name__ == "__main__":
    main()
