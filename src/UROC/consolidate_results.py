#!/usr/bin/env python3
"""
Consolidate per-experiment result directories into single .npz files.

Uses memory-mapped reads (mmap_mode='r') so only the last N_TERMINAL
columns are actually pulled from disk, avoiding full-file I/O.

Produces:
  results/direct_results.npz   — direct democracy full grid
  results/rep_results.npz      — representative democracy grid
"""

import os
import time
import numpy as np

N_TERMINAL = 10  # average last N timesteps for terminal values

METHODS = ["approval", "borda", "irv", "minimax", "plurality",
           "random_dictator", "star", "total_score"]

STATS = ["mean_history", "variance_history", "max_history", "min_history"]
STAT_KEYS = ["mean", "variance", "max", "min"]


def _load_terminal(path):
    """Load npy via mmap, return mean of last N_TERMINAL columns."""
    arr = np.load(path, mmap_mode='r')  # no copy into RAM
    return float(np.mean(arr[:, -N_TERMINAL:]))


# ──────────────────────────────────────────────────────────────────────
# Direct democracy
# ──────────────────────────────────────────────────────────────────────
def consolidate_direct():
    base = "results/direct"
    K_list     = list(range(1, 21))
    alpha_list = [round(a * 0.05, 2) for a in range(21)]
    nK, nA, nM = len(K_list), len(alpha_list), len(METHODS)

    arrays = {s: np.full((nK, nA, nM), np.nan) for s in STAT_KEYS}

    loaded = 0
    missing = 0
    t0 = time.time()
    for i, K in enumerate(K_list):
        for j, a in enumerate(alpha_list):
            for k, m in enumerate(METHODS):
                d = os.path.join(base, f"K{K}_a{a:.2f}", m)
                try:
                    for stat_file, stat_key in zip(STATS, STAT_KEYS):
                        p = os.path.join(d, f"{stat_file}.npy")
                        arrays[stat_key][i, j, k] = _load_terminal(p)
                    loaded += 1
                except FileNotFoundError:
                    missing += 1
        elapsed = time.time() - t0
        rate = loaded * 4 / elapsed if elapsed > 0 else 0
        print(f"  K={K:2d} done  ({loaded} configs, {rate:.0f} files/s)", flush=True)

    out = os.path.join("results", "direct_results.npz")
    np.savez_compressed(out,
                        K=np.array(K_list),
                        alpha=np.array(alpha_list),
                        methods=np.array(METHODS),
                        **arrays)
    print(f"Direct: loaded {loaded}, missing {missing}, "
          f"{time.time()-t0:.0f}s → {out}")


# ──────────────────────────────────────────────────────────────────────
# Representative democracy
# ──────────────────────────────────────────────────────────────────────
def consolidate_rep():
    base = "results/rep"
    K_list     = list(range(1, 21))
    alpha_list = [round(a * 0.1, 1) for a in range(11)]
    beta_list  = [0.0, 0.5, 1.0]
    pself_list = [0.0, 0.5, 1.0]
    nK, nA, nB, nP, nM = (len(K_list), len(alpha_list),
                           len(beta_list), len(pself_list), len(METHODS))

    arrays = {s: np.full((nK, nA, nB, nP, nM), np.nan) for s in STAT_KEYS}

    loaded = 0
    missing = 0
    t0 = time.time()
    for i, K in enumerate(K_list):
        for j, a in enumerate(alpha_list):
            for bi, b in enumerate(beta_list):
                for pi, p in enumerate(pself_list):
                    for k, m in enumerate(METHODS):
                        d = os.path.join(base,
                                         f"K{K}_a{a:.2f}",
                                         f"b{b:.1f}_p{p:.1f}",
                                         m)
                        try:
                            for stat_file, stat_key in zip(STATS, STAT_KEYS):
                                fp = os.path.join(d, f"{stat_file}.npy")
                                arrays[stat_key][i, j, bi, pi, k] = _load_terminal(fp)
                            loaded += 1
                        except FileNotFoundError:
                            missing += 1
        elapsed = time.time() - t0
        rate = loaded * 4 / elapsed if elapsed > 0 else 0
        print(f"  K={K:2d} done  ({loaded} configs, {rate:.0f} files/s)", flush=True)

    out = os.path.join("results", "rep_results.npz")
    np.savez_compressed(out,
                        K=np.array(K_list),
                        alpha=np.array(alpha_list),
                        beta=np.array(beta_list),
                        p_self=np.array(pself_list),
                        methods=np.array(METHODS),
                        **arrays)
    print(f"Rep: loaded {loaded}, missing {missing}, "
          f"{time.time()-t0:.0f}s → {out}")


def consolidate_scoring_fine():
    """Consolidate scoring_fine sweep: p × K × alpha, single 'generalized_borda' method."""
    p_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
              2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    K_list = list(range(1, 21))
    alpha_list = [round(a * 0.05, 2) for a in range(21)]
    nP, nK, nA = len(p_list), len(K_list), len(alpha_list)

    sf_stats = ["mean_history", "variance_history"]
    sf_keys  = ["mean", "variance"]
    arrays = {sk: np.full((nP, nK, nA), np.nan) for sk in sf_keys}
    loaded, missing = 0, 0
    t0 = time.time()

    for pi, p in enumerate(p_list):
        for i, K in enumerate(K_list):
            for j, alpha in enumerate(alpha_list):
                d = os.path.join("results", "scoring_fine",
                                 f"K{K}_a{alpha:.2f}", f"p{p:.2f}")
                for stat_file, stat_key in zip(sf_stats, sf_keys):
                    fp = os.path.join(d, f"{stat_file}.npy")
                    try:
                        arrays[stat_key][pi, i, j] = _load_terminal(fp)
                        loaded += 1
                    except FileNotFoundError:
                        missing += 1
        elapsed = time.time() - t0
        rate = loaded * 4 / elapsed if elapsed > 0 else 0
        print(f"  p={p:.2f} done  ({loaded} configs, {rate:.0f} files/s)", flush=True)

    out = os.path.join("results", "scoring_fine_results.npz")
    np.savez_compressed(out,
                        p=np.array(p_list),
                        K=np.array(K_list),
                        alpha=np.array(alpha_list),
                        **arrays)
    print(f"Scoring fine: loaded {loaded}, missing {missing}, "
          f"{time.time()-t0:.0f}s → {out}")


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["direct", "rep", "scoring_fine"]
    if "direct" in targets:
        print("=== Direct democracy ===", flush=True)
        consolidate_direct()
    if "rep" in targets:
        print("\n=== Representative democracy ===", flush=True)
        consolidate_rep()
    if "scoring_fine" in targets:
        print("\n=== Scoring fine ===", flush=True)
        consolidate_scoring_fine()
    print("\nDone.")
