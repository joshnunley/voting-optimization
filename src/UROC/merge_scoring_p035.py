#!/usr/bin/env python3
"""Merge scoring_p035 rep results into existing rep_results.npz."""

import os, time, numpy as np

N_TERMINAL = 10

def _load_terminal(path):
    arr = np.load(path, mmap_mode='r')
    return float(np.mean(arr[:, -N_TERMINAL:]))

# Load existing rep results
d = np.load('results/rep_results.npz')
K_list = list(d['K'])
A_list = list(d['alpha'])
B_list = list(d['beta'])
P_list = list(d['p_self'])
old_methods = list(d['methods'])
print(f'Existing: shape={d["mean"].shape}, methods={old_methods}')

# Load scoring_p035 data
base = 'results/rep'
STATS = ['mean_history', 'variance_history', 'max_history', 'min_history']
STAT_KEYS = ['mean', 'variance', 'max', 'min']
nK, nA, nB, nP = len(K_list), len(A_list), len(B_list), len(P_list)

new_arrays = {s: np.full((nK, nA, nB, nP), np.nan) for s in STAT_KEYS}
loaded, missing = 0, 0
t0 = time.time()

for i, K in enumerate(K_list):
    for j, a in enumerate(A_list):
        for bi, b in enumerate(B_list):
            for pi, p in enumerate(P_list):
                d_path = os.path.join(base, f'K{int(K)}_a{a:.2f}',
                                      f'b{b:.1f}_p{p:.1f}', 'scoring_p035')
                try:
                    for stat_file, stat_key in zip(STATS, STAT_KEYS):
                        fp = os.path.join(d_path, f'{stat_file}.npy')
                        new_arrays[stat_key][i, j, bi, pi] = _load_terminal(fp)
                    loaded += 1
                except FileNotFoundError:
                    missing += 1
    elapsed = time.time() - t0
    rate = loaded * 4 / elapsed if elapsed > 0 else 0
    print(f'  K={int(K):2d} done  ({loaded} configs, {rate:.0f} files/s)', flush=True)

print(f'scoring_p035: loaded {loaded}, missing {missing}, {time.time()-t0:.0f}s')

# Merge: append scoring_p035 as 9th method
new_methods = old_methods + ['scoring_p035']
merged = {}
for sk in STAT_KEYS:
    merged[sk] = np.concatenate([d[sk], new_arrays[sk][..., np.newaxis]], axis=-1)

out = 'results/rep_results_with_s035.npz'
np.savez_compressed(out,
    K=np.array(K_list),
    alpha=np.array(A_list),
    beta=np.array(B_list),
    p_self=np.array(P_list),
    methods=np.array(new_methods),
    **merged)
print(f'Saved {out}, shape={merged["mean"].shape}, methods={new_methods}')
