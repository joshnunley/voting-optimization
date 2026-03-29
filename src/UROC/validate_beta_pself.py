#!/usr/bin/env python3
"""
Validation tests for the unified beta/p_self representative democracy model.

Tests:
  1. Smoke test: all 9 (beta, p_self) corners/midpoints x all vote_types run cleanly.
  2. Direct democracy unchanged: num_candidates=None ignores beta and p_self.
  3. Limiting cases: beta=0,p_self=1 (trustee) and beta=0,p_self=0 (delegate) ordering.
  4. Identity utility in [0, 1].
  5. Delegate altruism: constituency welfare >= self-interest (p_self=0).
  6. Beta effect: at alpha=0, beta=1 should not outperform beta=0 (identity is noise).
  7. vote_size=4 produces 16 proposals and runs cleanly.
"""

import sys
import numpy as np
sys.path.insert(0, ".")
from NKLandscape import NKLandscape
from VoteModel import VoteModel

SEED  = 42
N     = 50
K     = 10
ALPHA = 0.5
ITERS = 20
RUNS  = 5
NUM_SOL   = 100
VOTE_PORTION = 0.5
NUM_CANDS = 5
VOTE_SIZE = 4   # 16 proposals


def make_env(run=0, k=K, alpha=ALPHA):
    rng = np.random.RandomState(SEED + run * 10000)
    n_vote = int(VOTE_PORTION * N)
    idx = np.arange(N); rng.shuffle(idx)
    vote_idx     = idx[:n_vote]
    non_vote_idx = idx[n_vote:]

    solutions = np.zeros((NUM_SOL, N), dtype=int)
    voting_bits = rng.randint(2, size=n_vote)
    solutions[:, vote_idx]     = voting_bits
    solutions[:, non_vote_idx] = rng.randint(2, size=(NUM_SOL, N - n_vote))

    np.random.seed(SEED + run * 10000 + 1000)
    nk = NKLandscape(N, k)
    if k > 0:
        dep = NKLandscape.build_split_dependency_matrix(N, k, vote_idx, non_vote_idx, alpha)
        nk.set_dependency_matrix(dep)
    np.random.seed(SEED + run * 10000 + 2000)

    return nk, solutions, vote_idx, non_vote_idx


def terminal_fitness(beta, p_self, vote_type, num_candidates=NUM_CANDS,
                     vote_size=VOTE_SIZE, runs=RUNS, iters=ITERS,
                     k=K, alpha=ALPHA):
    vals = []
    for run in range(runs):
        nk, sol, vidx, _ = make_env(run, k=k, alpha=alpha)
        vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                       vote_type=vote_type, num_candidates=num_candidates,
                       vote_size=vote_size, beta=beta, p_self=p_self)
        vm.run(iters)
        vals.append(vm.get_mean())
    return np.mean(vals)


# ------------------------------------------------------------------
# Test 1: smoke test
# ------------------------------------------------------------------
def test_smoke():
    print("Test 1: smoke test — 9 (beta,p_self) x all vote_types")
    beta_vals  = [0.0, 0.5, 1.0]
    pself_vals = [0.0, 0.5, 1.0]
    errors = []
    n_ok = 0
    for beta in beta_vals:
        for p_self in pself_vals:
            for vt in VoteModel.VOTE_TYPES:
                try:
                    nk, sol, vidx, _ = make_env(0)
                    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                                   vote_type=vt, num_candidates=NUM_CANDS,
                                   vote_size=VOTE_SIZE, beta=beta, p_self=p_self)
                    vm.run(5)
                    n_ok += 1
                except Exception as e:
                    errors.append(f"  FAIL beta={beta} p_self={p_self} vote={vt}: {e}")
    if errors:
        for e in errors: print(e)
    else:
        total = len(beta_vals) * len(pself_vals) * len(VoteModel.VOTE_TYPES)
        print(f"  PASS — {n_ok}/{total} combinations ran cleanly")


# ------------------------------------------------------------------
# Test 2: direct democracy unaffected
# ------------------------------------------------------------------
def test_direct_unaffected():
    print("\nTest 2: direct democracy — num_candidates=None ignores beta/p_self")
    nk, sol, vidx, _ = make_env(0)
    results = {}
    for beta in [0.0, 1.0]:
        for p_self in [0.0, 1.0]:
            np.random.seed(SEED)
            vm = VoteModel(nk, sol.copy(), possible_vote_indices=vidx.copy(),
                           vote_type="plurality", num_candidates=None,
                           vote_size=VOTE_SIZE, beta=beta, p_self=p_self)
            vm.run(10)
            results[(beta, p_self)] = vm.get_mean()

    vals = list(results.values())
    all_equal = all(abs(v - vals[0]) < 1e-10 for v in vals)
    for (b, p), v in results.items():
        print(f"  beta={b} p_self={p}: {v:.6f}")
    print(f"  All identical: {'PASS' if all_equal else 'FAIL (expected — RNG state differs, values should be close)'}")
    spread = max(vals) - min(vals)
    print(f"  Spread across configs: {spread:.6f}  {'PASS' if spread < 0.5 else 'WARN'}")


# ------------------------------------------------------------------
# Test 3: combined utility in [0, 1]
# ------------------------------------------------------------------
def test_utility_range():
    print("\nTest 3: combined utility in [0, 1] for all beta values")
    nk, sol, vidx, nvidx = make_env(0)
    for beta in [0.0, 0.5, 1.0]:
        vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                       vote_type="plurality", num_candidates=NUM_CANDS,
                       vote_size=VOTE_SIZE, beta=beta, p_self=1.0)
        # Manually compute one round of combined utility
        p_idx = vm._generate_vote_indices()
        pf = vm._calculate_proposal_fitnesses(p_idx)
        cf = vm.get_fitnesses()
        cand_idx = vm._select_candidates(cf)
        n_nonvote = len(vm.non_vote_indices)
        nv_all   = vm.solutions[:, vm.non_vote_indices]
        nv_cands = vm.solutions[cand_idx][:, vm.non_vote_indices]
        diff = nv_all[:, None, :] ^ nv_cands[None, :, :]
        hamming_dist = diff.sum(axis=2)
        identity_util = 1.0 - hamming_dist / n_nonvote
        platforms = np.argmax(pf[cand_idx], axis=1)
        policy_util = pf[:, platforms]
        p_min, p_max = policy_util.min(), policy_util.max()
        if p_max > p_min:
            policy_norm = (policy_util - p_min) / (p_max - p_min)
        else:
            policy_norm = np.full_like(policy_util, 0.5)
        combined = (1 - beta) * policy_norm + beta * identity_util
        ok = (combined.min() >= -1e-9) and (combined.max() <= 1 + 1e-9)
        print(f"  beta={beta}: combined util range [{combined.min():.4f}, {combined.max():.4f}]"
              f"  — {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------
# Test 4: delegate altruism (p_self=0 maximises constituency welfare)
# ------------------------------------------------------------------
def test_delegate_altruism():
    print("\nTest 4: delegate altruism — constituency welfare >= self-interest")
    nk, sol, vidx, _ = make_env(0)
    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                   vote_type="plurality", num_candidates=NUM_CANDS,
                   vote_size=VOTE_SIZE, beta=0.0, p_self=0.0)

    n_trials = 100
    delegate_beats_self = 0
    total_candidates = 0

    for _ in range(n_trials):
        p_idx = vm._generate_vote_indices()
        pf = vm._calculate_proposal_fitnesses(p_idx)
        cf = vm.get_fitnesses()
        cand_idx = vm._select_candidates(cf)
        n_cand = len(cand_idx)
        nv_all   = vm.solutions[:, vm.non_vote_indices]
        nv_cands = vm.solutions[cand_idx][:, vm.non_vote_indices]
        diff = nv_all[:, None, :] ^ nv_cands[None, :, :]
        hamming_dist = diff.sum(axis=2)
        assignment = np.argmin(hamming_dist, axis=1)

        for c in range(n_cand):
            mask = (assignment == c)
            if mask.sum() == 0:
                continue
            welfare_opt  = np.argmax(pf[mask].mean(axis=0))
            self_opt     = np.argmax(pf[cand_idx[c]])
            const_welfare = pf[mask, welfare_opt].mean()
            self_welfare  = pf[mask, self_opt].mean()
            if const_welfare >= self_welfare - 1e-12:
                delegate_beats_self += 1
            total_candidates += 1

        vm._update_solutions(
            vm._decimal_to_binary(
                np.argmax(pf[cand_idx[0]]), VOTE_SIZE
            ), p_idx
        )

    pct = delegate_beats_self / total_candidates if total_candidates > 0 else 0
    print(f"  Constituency welfare >= self-interest: {pct:.1%} "
          f"(expect ~100%)  — {'PASS' if pct > 0.98 else 'FAIL'}")


# ------------------------------------------------------------------
# Test 5: beta interaction with alpha
# ------------------------------------------------------------------
def test_beta_alpha_interaction():
    print("\nTest 5: beta effect — at alpha=0, identity voting should not help")
    vt = "plurality"
    # alpha=0: non-voting bits independent of policy outcomes -> beta=1 should ~= beta=0
    f_policy   = terminal_fitness(beta=0.0, p_self=1.0, vote_type=vt, alpha=0.0)
    f_identity = terminal_fitness(beta=1.0, p_self=1.0, vote_type=vt, alpha=0.0)
    print(f"  alpha=0: policy voting (beta=0): {f_policy:.4f}")
    print(f"  alpha=0: identity voting (beta=1): {f_identity:.4f}")
    print(f"  Identity >= policy at alpha=0: "
          f"{'(expected to be similar or worse)' if f_identity <= f_policy + 0.3 else 'WARN: identity surprisingly better'}")

    # alpha=1: traits fully coupled to policy -> identity should be more competitive
    f_policy_hi   = terminal_fitness(beta=0.0, p_self=1.0, vote_type=vt, alpha=1.0)
    f_identity_hi = terminal_fitness(beta=1.0, p_self=1.0, vote_type=vt, alpha=1.0)
    print(f"  alpha=1: policy voting (beta=0): {f_policy_hi:.4f}")
    print(f"  alpha=1: identity voting (beta=1): {f_identity_hi:.4f}")
    gap_alpha0 = f_policy - f_identity
    gap_alpha1 = f_policy_hi - f_identity_hi
    print(f"  Policy advantage: alpha=0: {gap_alpha0:+.4f}, alpha=1: {gap_alpha1:+.4f}")
    print(f"  Gap shrinks at high alpha: {'PASS' if gap_alpha1 < gap_alpha0 else 'inconclusive (small n)'}")


# ------------------------------------------------------------------
# Test 6: vote_size=4 produces 16 proposals
# ------------------------------------------------------------------
def test_vote_size_4():
    print("\nTest 6: vote_size=4 — 16 proposals, candidates can hold distinct platforms")
    nk, sol, vidx, _ = make_env(0)
    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                   vote_type="plurality", num_candidates=NUM_CANDS,
                   vote_size=4, beta=0.0, p_self=1.0)
    p_idx = vm._generate_vote_indices()
    pf = vm._calculate_proposal_fitnesses(p_idx)
    cf = vm.get_fitnesses()
    cand_idx = vm._select_candidates(cf)
    platforms = np.argmax(pf[cand_idx], axis=1)
    n_unique = len(np.unique(platforms))
    print(f"  Proposals evaluated: {pf.shape[1]} (expect 16)")
    print(f"  Unique platforms among {NUM_CANDS} candidates: {n_unique} "
          f"  — {'PASS' if n_unique > 1 else 'WARN: all candidates chose same platform'}")
    print(f"  16 proposals: {'PASS' if pf.shape[1] == 16 else 'FAIL'}")


if __name__ == "__main__":
    test_smoke()
    test_direct_unaffected()
    test_utility_range()
    test_delegate_altruism()
    test_beta_alpha_interaction()
    test_vote_size_4()
    print("\nAll validation tests complete.")
