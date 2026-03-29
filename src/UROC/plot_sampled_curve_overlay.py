#!/usr/bin/env python3
"""
Overlay a sparsely sampled candidate-method rank-vs-complexity curve on top of
the existing 8-method baseline plot.

Baseline methods:
  - use the consolidated `results/direct_results.npz`
  - compute rank-vs-K0 curves using the fixed paper formula and Gaussian
    weighting in K around each iso-complexity parabola

Candidate method:
  - load sampled point data from a single .npz produced by
    `generate_sampled_curve_data.py`
  - at each sampled (K, alpha), interpolate the 8 baseline methods in alpha
    and compute the candidate's rank against them
  - average those ranks within each K0
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sample_complexity_curve import complexity_params


BASELINE_COLORS = {
    "approval": "#e41a1c",
    "borda": "#377eb8",
    "irv": "#4daf4a",
    "minimax": "#984ea3",
    "plurality": "#ff7f00",
    "random_dictator": "#a65628",
    "star": "#f781bf",
    "total_score": "#999999",
}

BASELINE_LABELS = {
    "approval": "Approval",
    "borda": "Borda",
    "irv": "IRV",
    "minimax": "Minimax",
    "plurality": "Plurality",
    "random_dictator": "Rand. Dict.",
    "star": "STAR",
    "total_score": "Total Score",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_npz", required=True,
                   help="Sample data .npz generated for the candidate method")
    p.add_argument("--baseline_npz", default="results/direct_results.npz",
                   help="Consolidated direct democracy baseline results")
    p.add_argument("--output", required=True,
                   help="Output plot path")
    p.add_argument("--eps", type=float, default=0.5,
                   help="Gaussian K-distance bandwidth for baseline curves")
    p.add_argument("--candidate_label", default=None,
                   help="Optional legend label override for candidate curve")
    p.add_argument("--candidate_color", default="#111111",
                   help="Color for candidate curve")
    return p.parse_args()


def compute_baseline_ranks(mean):
    ranks = np.zeros_like(mean)
    nK, nA, nM = mean.shape
    for i in range(nK):
        for j in range(nA):
            order = np.argsort(-mean[i, j, :])
            for r, idx in enumerate(order):
                ranks[i, j, idx] = r + 1
    return ranks


def compute_fixed_formula_baseline_curves(K_arr, alpha_arr, ranks_full, eps):
    methods_n = ranks_full.shape[2]
    K0_values = np.arange(1, 21, dtype=float)
    curves = np.zeros((len(K0_values), methods_n), dtype=float)

    for ki, K0 in enumerate(K0_values):
        c_val, mu_val = complexity_params(K0)
        total_w = 0.0
        w_rank = np.zeros(methods_n, dtype=float)
        for j, alpha in enumerate(alpha_arr):
            K_on_curve = c_val * (alpha - mu_val) ** 2 + K0
            for i, K_cell in enumerate(K_arr):
                dist = abs(K_cell - K_on_curve)
                w = np.exp(-(dist ** 2) / (2.0 * eps ** 2))
                w_rank += w * ranks_full[i, j, :]
                total_w += w
        curves[ki] = w_rank / total_w

    return K0_values, curves


def interp_baseline_means_at_alpha(mean_full, K_arr, alpha_arr, K, alpha):
    k_idx = int(np.where(K_arr == int(K))[0][0])
    out = np.zeros(mean_full.shape[2], dtype=float)
    for m in range(mean_full.shape[2]):
        out[m] = np.interp(alpha, alpha_arr, mean_full[k_idx, :, m])
    return out


def compute_candidate_rank_curve(sample_data, baseline_data):
    mean_full = baseline_data["mean"]
    K_arr = baseline_data["K"]
    alpha_arr = baseline_data["alpha"]

    sampled_counts = sample_data["sampled_counts"]
    sampled_K = sample_data["sampled_K"]
    sampled_alpha = sample_data["sampled_alpha"]
    sampled_terminal_mean = sample_data["sampled_terminal_mean"]
    K0_values = sample_data["K0"].astype(float)

    candidate_mean_rank = np.full(len(K0_values), np.nan, dtype=float)
    candidate_rank_se = np.full(len(K0_values), np.nan, dtype=float)

    for i, K0 in enumerate(K0_values):
        point_ranks = []
        for j in range(int(sampled_counts[i])):
            K = sampled_K[i, j]
            alpha = sampled_alpha[i, j]
            cand_mean = sampled_terminal_mean[i, j]
            if np.isnan(K) or np.isnan(alpha) or np.isnan(cand_mean):
                continue
            baseline_means = interp_baseline_means_at_alpha(mean_full, K_arr, alpha_arr, int(round(K)), float(alpha))
            joint = np.concatenate([baseline_means, [cand_mean]])
            order = np.argsort(-joint)
            candidate_rank = int(np.where(order == len(joint) - 1)[0][0]) + 1
            point_ranks.append(candidate_rank)

        if point_ranks:
            point_ranks = np.array(point_ranks, dtype=float)
            candidate_mean_rank[i] = float(np.mean(point_ranks))
            candidate_rank_se[i] = float(np.std(point_ranks) / np.sqrt(len(point_ranks)))

    return K0_values, candidate_mean_rank, candidate_rank_se


def main():
    args = parse_args()
    sample = np.load(args.sample_npz, allow_pickle=True)
    baseline = np.load(args.baseline_npz, allow_pickle=True)

    mean_full = baseline["mean"]
    methods = list(baseline["methods"])
    ranks_full = compute_baseline_ranks(mean_full)
    K0_baseline, baseline_curves = compute_fixed_formula_baseline_curves(
        baseline["K"], baseline["alpha"], ranks_full, eps=args.eps
    )

    K0_candidate, candidate_curve, candidate_se = compute_candidate_rank_curve(sample, baseline)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_baseline = K0_baseline / 20.0
    for midx, method in enumerate(methods):
        ax.plot(
            x_baseline,
            baseline_curves[:, midx],
            color=BASELINE_COLORS.get(method, None),
            linewidth=2.2,
            label=BASELINE_LABELS.get(method, method),
        )

    x_candidate = K0_candidate / 20.0
    if args.candidate_label is not None:
        candidate_label = args.candidate_label
    else:
        candidate_label = sample["vote_type"].item() if hasattr(sample["vote_type"], "item") else str(sample["vote_type"])
    valid = ~np.isnan(candidate_curve)
    ax.plot(
        x_candidate[valid],
        candidate_curve[valid],
        color=args.candidate_color,
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        linestyle="-",
        label=candidate_label + " (sampled)",
    )
    if np.any(valid):
        y0 = candidate_curve[valid] - candidate_se[valid]
        y1 = candidate_curve[valid] + candidate_se[valid]
        ax.fill_between(x_candidate[valid], y0, y1, color=args.candidate_color, alpha=0.12)

    ax.set_xlabel("Complexity (K0 / 20)", fontsize=13)
    ax.set_ylabel("Mean rank", fontsize=13)
    ax.set_title(
        "Voting Method Rank vs Complexity\n"
        "Baseline methods from full grid; candidate curve ranked against the 8 baselines at sampled points",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    ax.set_yticks(range(1, 10))
    ax.legend(fontsize=10, ncol=2, loc="center right")

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
