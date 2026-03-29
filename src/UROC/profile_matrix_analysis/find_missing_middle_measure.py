#!/usr/bin/env python3
"""
Search for a single profile-level measure that peaks in the Borda-dominant
middle regime, roughly K0 in [8, 15].

This complements the earlier regime triple search, which already found good
curves for:
  - trivial early simplicity
  - high-complexity fragmentation

Here we specifically target the missing middle feature. To make that feasible,
we search not only primitive normalized features and simple transforms, but also
small composite curves built from pairs of normalized feature curves.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_RANGE = (8, 15)


def parse_args():
    p = argparse.ArgumentParser(description="Search for a missing middle-regime measure")
    p.add_argument(
        "--input_npz",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/raw_profile_matrix_analysis.npz",
        help="Input feature dataset from analyze_raw_profile_matrices.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="src/UROC/results/profile_matrix_analysis/raw_u_regime_scan/missing_middle_measure",
        help="Directory for outputs",
    )
    p.add_argument(
        "--max_feature_pool",
        type=int,
        default=15,
        help="Restrict the primitive feature pool to the top-N candidate features from the clustering pass",
    )
    p.add_argument("--top_n", type=int, default=75)
    return p.parse_args()


def normalize_curve(y):
    y = np.asarray(y, dtype=float)
    lo = np.min(y)
    hi = np.max(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def target_mask(k0_values):
    k0 = np.asarray(k0_values, dtype=int)
    return (k0 >= TARGET_RANGE[0]) & (k0 <= TARGET_RANGE[1])


def specialization_score(curve, mask):
    return float(np.mean(curve[mask]) - np.mean(curve[~mask]))


def peak_location_score(curve, k0_values):
    k0_values = np.asarray(k0_values, dtype=int)
    peak_k0 = int(k0_values[np.argmax(curve)])
    center = 0.5 * (TARGET_RANGE[0] + TARGET_RANGE[1])
    return -abs(peak_k0 - center)


def read_top_feature_pool(input_npz: Path, max_feature_pool: int):
    top_csv = input_npz.parent / "clustering" / "top_discriminative_features.csv"
    if top_csv.exists():
        rows = list(csv.DictReader(top_csv.open()))
        feats = [r["feature"] for r in rows[:max_feature_pool]]
        if feats:
            return feats
    return None


def primitive_curves(feature_names, mean_by_k0):
    curves = {}
    for name, y in zip(feature_names, mean_by_k0.T):
        x = normalize_curve(y)
        curves[name] = x
        curves[f"inv({name})"] = 1.0 - x
        curves[f"bell({name})"] = 4.0 * x * (1.0 - x)
    return curves


def composite_curves(base_curves):
    names = list(base_curves.keys())
    out = {}
    for a, b in itertools.combinations(names, 2):
        xa = base_curves[a]
        xb = base_curves[b]
        out[f"{a}*{b}"] = normalize_curve(xa * xb)
        out[f"{a}*inv({b})"] = normalize_curve(xa * (1.0 - xb))
        out[f"inv({a})*{b}"] = normalize_curve((1.0 - xa) * xb)
        out[f"bell({a})*{b}"] = normalize_curve((4.0 * xa * (1.0 - xa)) * xb)
        out[f"{a}*bell({b})"] = normalize_curve(xa * (4.0 * xb * (1.0 - xb)))
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_top_curves(k0_values, curves, names, outpath):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00"]
    for color, name in zip(colors, names):
        ax.plot(k0_values, curves[name], label=name, linewidth=2.0, color=color)
    ax.axvspan(TARGET_RANGE[0], TARGET_RANGE[1], color="#4daf4a", alpha=0.10)
    ax.set_xlabel("K0")
    ax.set_ylabel("normalized value")
    ax.set_title("Top candidate measures for the missing middle regime")
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

    base = primitive_curves(feature_names, mean_by_k0)
    composites = composite_curves({k: v for k, v in base.items() if not k.startswith("inv(") and not k.startswith("bell(")})
    curves = {}
    curves.update(base)
    curves.update(composites)

    mask = target_mask(k0_values)
    rows = []
    for name, curve in curves.items():
        score = specialization_score(curve, mask)
        peak_bonus = peak_location_score(curve, k0_values)
        rows.append({
            "curve": name,
            "specialization_score": float(score),
            "peak_location_bonus": float(peak_bonus),
            "combined_score": float(score + 0.02 * peak_bonus),
            "peak_k0": int(k0_values[np.argmax(curve)]),
        })

    rows.sort(key=lambda r: (-r["combined_score"], -r["specialization_score"], abs(r["peak_k0"] - int(np.mean(TARGET_RANGE))), r["curve"]))
    write_csv(
        outdir / "top_middle_measures.csv",
        rows[:args.top_n],
        ["curve", "specialization_score", "peak_location_bonus", "combined_score", "peak_k0"],
    )

    best_names = [r["curve"] for r in rows[:5]]
    plot_top_curves(k0_values, curves, best_names, outdir / "top_middle_measures.pdf")

    np.savez_compressed(
        outdir / "missing_middle_measure_search.npz",
        k0_values=k0_values,
        top_curve_names=np.array(best_names, dtype=object),
        top_curve_matrix=np.vstack([curves[n] for n in best_names]),
        best_curve=np.array(rows[0]["curve"], dtype=object),
        best_score=np.array(rows[0]["combined_score"]),
    )

    print("Top 10 candidate middle measures:")
    for row in rows[:10]:
        print(
            f"  {row['curve']} :: spec={row['specialization_score']:.3f}, "
            f"peakK0={row['peak_k0']}, combined={row['combined_score']:.3f}"
        )
    print(f"saved {outdir / 'missing_middle_measure_search.npz'}")


if __name__ == "__main__":
    main()
