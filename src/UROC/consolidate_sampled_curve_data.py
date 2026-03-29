#!/usr/bin/env python3
"""
Consolidate multiple sampled-curve .npz files produced by
generate_sampled_curve_data.py into a single aggregated .npz.

This keeps the same schema expected by plot_sampled_curve_overlay.py:
the output stores aggregated terminal_mean / terminal_variance and their SEs.
"""

import argparse
import os
import numpy as np


META_KEYS = [
    "vote_type",
    "K0",
    "num_points_requested",
    "curve_a",
    "curve_b",
    "scoring_power_p",
    "terminal_window",
    "iterations_base",
    "iterations_slope",
    "sampled_counts",
    "sampled_K",
    "sampled_alpha",
    "sampled_arc_fraction",
    "sampled_iterations",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob", required=True,
                   help="Glob pattern for per-seed sampled .npz files")
    p.add_argument("--output_npz", required=True,
                   help="Output consolidated .npz path")
    return p.parse_args()


def main():
    args = parse_args()

    import glob
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise ValueError("No input files matched {}".format(args.input_glob))

    loaded = [np.load(f, allow_pickle=True) for f in files]

    ref = loaded[0]
    for arr in loaded[1:]:
        for key in ["sampled_counts", "sampled_K", "sampled_alpha", "sampled_arc_fraction"]:
            if not np.allclose(ref[key], arr[key], equal_nan=True):
                raise ValueError("Input sampled grids do not match for key {}".format(key))

    mean_stack = np.stack([arr["sampled_terminal_mean"] for arr in loaded], axis=0)
    var_stack = np.stack([arr["sampled_terminal_variance"] for arr in loaded], axis=0)

    out = {}
    for key in META_KEYS:
        out[key] = ref[key]

    out["runs"] = np.array(len(files))
    out["source_files"] = np.array(files)
    out["sampled_terminal_mean"] = np.nanmean(mean_stack, axis=0)
    out["sampled_terminal_variance"] = np.nanmean(var_stack, axis=0)

    mean_n = np.sum(~np.isnan(mean_stack), axis=0)
    var_n = np.sum(~np.isnan(var_stack), axis=0)

    mean_sd = np.nanstd(mean_stack, axis=0)
    var_sd = np.nanstd(var_stack, axis=0)
    out["sampled_terminal_mean_se"] = np.where(mean_n > 0, mean_sd / np.sqrt(mean_n), np.nan)
    out["sampled_terminal_variance_se"] = np.where(var_n > 0, var_sd / np.sqrt(var_n), np.nan)

    outdir = os.path.dirname(args.output_npz)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    print("saved {}".format(args.output_npz), flush=True)


if __name__ == "__main__":
    main()
