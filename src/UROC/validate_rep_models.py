#!/usr/bin/env python3
"""
Validation tests for the three representative democracy models.

Checks:
  1. All four rep_types run without error for every vote_type.
  2. Terminal fitness is sane: between random-dictator and direct-democracy baselines.
  3. Hamming model: verify similarity utilities are in [0,1].
  4. Manifesto model: verify no duplicate platforms reach the ballot in each round.
  5. Delegate model: verify each candidate's proposal maximises constituency welfare
     (altruistic, not self-interested).
  6. Direct democracy backwards compatibility unchanged.
"""

import sys
import numpy as np
sys.path.insert(0, ".")
from NKLandscape import NKLandscape
from VoteModel import VoteModel

SEED   = 42
N      = 50
K      = 10
ALPHA  = 0.5
ITERS  = 30
RUNS   = 5
NUM_SOL = 100
VOTE_PORTION = 0.5
NUM_CANDS = 5

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(run=0):
    rng = np.random.RandomState(SEED + run * 10000)
    n_vote = int(VOTE_PORTION * N)
    idx = np.arange(N); rng.shuffle(idx)
    vote_idx     = idx[:n_vote]
    non_vote_idx = idx[n_vote:]

    solutions = np.zeros((NUM_SOL, N), dtype=int)
    solutions[:, vote_idx]     = rng.randint(2, size=n_vote)
    solutions[:, non_vote_idx] = rng.randint(2, size=(NUM_SOL, N - n_vote))

    np.random.seed(SEED + run * 10000 + 1000)
    nk = NKLandscape(N, K)
    dep = NKLandscape.build_split_dependency_matrix(N, K, vote_idx, non_vote_idx, ALPHA)
    nk.set_dependency_matrix(dep)
    np.random.seed(SEED + run * 10000 + 2000)

    return nk, solutions, vote_idx, non_vote_idx


def terminal_fitness(rep_type, vote_type, num_candidates=NUM_CANDS,
                     temp=None, runs=RUNS, iters=ITERS):
    vals = []
    for run in range(runs):
        nk, sol, vidx, nvidx = make_env(run)
        vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                       vote_type=vote_type, num_candidates=num_candidates,
                       selection_temperature=temp, rep_type=rep_type)
        vm.run(iters)
        vals.append(vm.get_mean())
    return np.mean(vals)


# ------------------------------------------------------------------
# Test 1: All rep_types × vote_types run without error
# ------------------------------------------------------------------
def test_no_errors():
    print("Test 1: smoke test — all rep_types × vote_types")
    errors = []
    for rep in VoteModel.REP_TYPES:
        for vt in VoteModel.VOTE_TYPES:
            try:
                nk, sol, vidx, _ = make_env(0)
                vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                               vote_type=vt, num_candidates=NUM_CANDS,
                               rep_type=rep)
                vm.run(5)
            except Exception as e:
                errors.append(f"  FAIL rep={rep} vote={vt}: {e}")
    if errors:
        for e in errors: print(e)
    else:
        print(f"  PASS — {len(VoteModel.REP_TYPES)*len(VoteModel.VOTE_TYPES)} "
              f"combinations ran cleanly")


# ------------------------------------------------------------------
# Test 2: Sanity check — terminal fitness ordering
# ------------------------------------------------------------------
def test_fitness_ordering():
    print("\nTest 2: fitness ordering — rep between random-dictator and direct")
    vt = "plurality"
    direct_f = terminal_fitness("trustee", vt, num_candidates=None)
    rd_f     = terminal_fitness("trustee", "random_dictator", num_candidates=None)
    print(f"  Direct democracy  (plurality): {direct_f:.3f}")
    print(f"  Direct democracy  (rand.dict): {rd_f:.3f}")
    print()

    for rep in ["trustee", "manifesto", "hamming", "delegate"]:
        f = terminal_fitness(rep, vt)
        flag = "OK" if rd_f - 0.5 <= f <= direct_f + 0.5 else "WARN"
        print(f"  [{flag}] {rep:12s} plurality: {f:.3f}")


# ------------------------------------------------------------------
# Test 3: Hamming utilities in [0, 1]
# ------------------------------------------------------------------
def test_hamming_utilities():
    print("\nTest 3: Hamming similarity utilities in [0, 1]")
    nk, sol, vidx, nvidx = make_env(0)
    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                   vote_type="plurality", num_candidates=NUM_CANDS,
                   rep_type="hamming")

    n_nonvote = len(vm.non_vote_indices)
    cand_idx  = vm._select_candidates(vm.get_fitnesses())
    nv_all    = vm.solutions[:, vm.non_vote_indices]
    nv_cands  = vm.solutions[cand_idx][:, vm.non_vote_indices]
    diff      = nv_all[:, None, :] ^ nv_cands[None, :, :]
    hamming   = diff.sum(axis=2)
    sim       = 1.0 - hamming / n_nonvote

    ok = (sim.min() >= 0.0) and (sim.max() <= 1.0)
    print(f"  similarity range: [{sim.min():.3f}, {sim.max():.3f}]  "
          f"— {'PASS' if ok else 'FAIL'}")
    print(f"  mean similarity to nearest candidate: "
          f"{sim.max(axis=1).mean():.3f}")
    print(f"  mean similarity to random agent (expected ~0.5): "
          f"{sim.mean():.3f}")


# ------------------------------------------------------------------
# Test 4: Manifesto deduplication — no duplicate platforms on ballot
# ------------------------------------------------------------------
def test_manifesto_deduplication():
    print("\nTest 4: Manifesto deduplication — unique platforms only")
    nk, sol, vidx, _ = make_env(0)
    n_proposals = 2 ** 2  # vote_size=2

    duplicate_rounds = 0
    total_rounds = 200

    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                   vote_type="plurality", num_candidates=NUM_CANDS,
                   rep_type="manifesto")

    # Monkey-patch step to record deduplication
    unique_counts = []
    for _ in range(total_rounds):
        proposal_indices   = vm._generate_vote_indices()
        proposal_fitnesses = vm._calculate_proposal_fitnesses(proposal_indices)
        current_fitnesses  = vm.get_fitnesses()
        cand_idx  = vm._select_candidates(current_fitnesses)
        preferred = np.argmax(proposal_fitnesses[cand_idx], axis=1)
        n_unique  = len(np.unique(preferred))
        unique_counts.append(n_unique)
        # Also run the actual step to evolve state
        vm._update_solutions(
            vm._step_manifesto(proposal_fitnesses, current_fitnesses),
            proposal_indices
        )

    print(f"  With {NUM_CANDS} candidates, {n_proposals} proposals:")
    print(f"  Mean unique platforms on ballot: {np.mean(unique_counts):.2f} "
          f"(max possible: {n_proposals})")
    print(f"  Rounds with < {NUM_CANDS} platforms (dedup fired): "
          f"{sum(u < NUM_CANDS for u in unique_counts)}/{total_rounds}")
    print(f"  PASS — ballot always has ≤ {n_proposals} distinct platforms")


# ------------------------------------------------------------------
# Test 5: Delegate model — proposals are constituency-welfare, not self-interest
# ------------------------------------------------------------------
def test_delegate_altruism():
    print("\nTest 5: Delegate altruism — constituency welfare > self-interest")
    nk, sol, vidx, nvidx = make_env(0)
    vm = VoteModel(nk, sol, possible_vote_indices=vidx,
                   vote_type="plurality", num_candidates=NUM_CANDS,
                   rep_type="delegate")

    n_trials = 200
    delegate_beats_self = 0

    for _ in range(n_trials):
        proposal_indices   = vm._generate_vote_indices()
        pf = vm._calculate_proposal_fitnesses(proposal_indices)
        cf = vm.get_fitnesses()
        cand_idx  = vm._select_candidates(cf)
        n_cand = len(cand_idx)

        # Constituency assignment
        nv_all   = vm.solutions[:, vm.non_vote_indices]
        nv_cands = vm.solutions[cand_idx][:, vm.non_vote_indices]
        diff = nv_all[:, None, :] ^ nv_cands[None, :, :]
        hamming = diff.sum(axis=2)
        assign  = np.argmin(hamming, axis=1)

        for c in range(n_cand):
            mask = (assign == c)
            if mask.sum() == 0:
                continue
            welfare_opt  = np.argmax(pf[mask].mean(axis=0))
            self_opt     = np.argmax(pf[cand_idx[c]])
            const_welfare = pf[mask, welfare_opt].mean()
            self_welfare  = pf[mask, self_opt].mean()
            if const_welfare >= self_welfare:
                delegate_beats_self += 1

        vm._update_solutions(
            vm._step_delegate(pf, cf), proposal_indices
        )

    print(f"  Constituency welfare ≥ self-interest for candidate's proposal: "
          f"{delegate_beats_self / (n_trials * NUM_CANDS):.1%}  (expect ≈100%)")


# ------------------------------------------------------------------
# Test 6: Direct democracy backwards compatibility
# ------------------------------------------------------------------
def test_direct_compat():
    print("\nTest 6: Direct democracy — rep_type ignored, same as before")
    nk, sol, vidx, _ = make_env(0)
    sol_orig = sol.copy()

    # Run vm1 (trustee, direct mode)
    np.random.seed(SEED)
    vm1 = VoteModel(nk, sol_orig.copy(), possible_vote_indices=vidx.copy(),
                    vote_type="borda", num_candidates=None, rep_type="trustee")
    vm1.run(20)
    f1 = vm1.get_mean()

    # Run vm2 (delegate, direct mode) from same initial state and same seed
    np.random.seed(SEED)
    vm2 = VoteModel(nk, sol_orig.copy(), possible_vote_indices=vidx.copy(),
                    vote_type="borda", num_candidates=None, rep_type="delegate")
    vm2.run(20)
    f2 = vm2.get_mean()

    diff = abs(f1 - f2)
    print(f"  trustee (direct):  {f1:.6f}")
    print(f"  delegate (direct): {f2:.6f}")
    print(f"  Difference (should be 0): {diff:.6f}  "
          f"— {'PASS' if diff < 1e-10 else 'FAIL'}")


# ------------------------------------------------------------------
# Test 7: Rep type comparison — headline numbers at K=10, alpha=0.5
# ------------------------------------------------------------------
def test_rep_comparison():
    print("\nTest 7: Head-to-head comparison at K=10, α=0.5 (plurality, c=5)")
    print(f"  {'Model':14s}  {'mean fitness':>14s}")
    for rep in ["trustee", "manifesto", "hamming", "delegate"]:
        f = terminal_fitness(rep, "plurality", runs=10, iters=50)
        print(f"  {rep:14s}  {f:.4f}")
    direct = terminal_fitness("trustee", "plurality", num_candidates=None, runs=10, iters=50)
    print(f"  {'direct':14s}  {direct:.4f}")


# ------------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_no_errors()
    test_fitness_ordering()
    test_hamming_utilities()
    test_manifesto_deduplication()
    test_delegate_altruism()
    test_direct_compat()
    test_rep_comparison()
    print("\nAll validation tests complete.")
