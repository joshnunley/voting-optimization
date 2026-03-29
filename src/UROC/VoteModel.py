import numpy as np


class VoteModel:
    """
    Collective decision-making model using voting methods on NK landscapes.

    Direct democracy (num_candidates=None):
      All agents vote directly on proposals each round.

    Representative democracy (num_candidates=C):
      C candidates are nominated, each announces a platform, the population
      elects one candidate, and the winner's platform is implemented.
      Controlled by two parameters:

      beta in [0, 1] — voter identity weight:
        beta=0 : voters evaluate candidates by policy outcomes
                 (what fitness would I get under each candidate's platform?)
        beta=1 : voters evaluate candidates by Hamming similarity in individual
                 traits (descriptive / identity-based representation)
        Mixed  : convex combination, both components normalised to [0, 1].

      p_self in [0, 1] — candidate selfishness:
        p_self=0 : each candidate proposes the policy that maximises their
                   constituency's mean welfare (delegate / altruistic)
        p_self=1 : each candidate proposes the policy that maximises their
                   own fitness (trustee / self-interested)
        Mixed    : each candidate independently draws Bernoulli(p_self) to
                   decide which objective to use.

    Classical limiting cases
    ------------------------
    beta=0, p_self=1  : trustee       — policy election, self-interested platform
    beta=0, p_self=0  : delegate      — policy election, constituency platform
    beta=1, p_self=1  : id-trustee    — identity election, self-interested platform
    beta=1, p_self=0  : id-delegate   — identity election, constituency platform

    Constituency assignment is always by nearest-neighbour Hamming distance on
    individual (non-voting) bits, regardless of beta or p_self.
    """

    VOTE_TYPES = [
        "plurality", "approval", "total_score",
        "borda", "irv", "star", "minimax", "random_dictator",
        "scoring_p035",
    ]

    def __init__(self, nk_model, solutions, possible_vote_indices=None,
                 vote_size=2, vote_type="plurality", num_candidates=None,
                 selection_temperature=None, beta=0.0, p_self=1.0):
        if nk_model is None:
            raise ValueError("An NKLandscape model must be provided")
        if solutions is None:
            raise ValueError("A set of voting solutions must be provided")

        self.nk = nk_model
        self.solutions = np.copy(solutions)
        self.num_solutions = solutions.shape[0]
        self.possible_vote_indices = (
            np.arange(self.nk.N) if possible_vote_indices is None
            else np.copy(possible_vote_indices)
        )
        all_indices = np.arange(self.nk.N)
        self.non_vote_indices = np.setdiff1d(all_indices, self.possible_vote_indices)

        self.vote_size = vote_size
        self.vote_type = vote_type
        self.num_candidates = num_candidates
        self.selection_temperature = selection_temperature
        self.beta = float(beta)
        self.p_self = float(p_self)

    # ------------------------------------------------------------------ #
    # Core voting loop                                                     #
    # ------------------------------------------------------------------ #

    def run(self, iterations=100, until_unique=False):
        for _ in range(iterations):
            self.step()
            if until_unique and self.num_solutions == 1:
                break

    def step(self):
        """Execute a single voting round."""
        proposal_indices   = self._generate_vote_indices()
        proposal_fitnesses = self._calculate_proposal_fitnesses(proposal_indices)
        current_fitnesses  = self.get_fitnesses()

        if self.num_candidates is None:
            winner_idx = self._vote(proposal_fitnesses, current_fitnesses)
            winner = self._decimal_to_binary(winner_idx, self.vote_size)
        else:
            winner = self._step_representative(proposal_fitnesses, current_fitnesses)

        self._update_solutions(winner, proposal_indices)

    # ------------------------------------------------------------------ #
    # Representative democracy                                             #
    # ------------------------------------------------------------------ #

    def _step_representative(self, proposal_fitnesses, current_fitnesses):
        """
        Unified representative democracy step.

        1. Nominate candidates (softmax or uniform).
        2. Assign every voter to their nearest candidate by Hamming distance
           on individual (non-voting) bits.
        3. Each candidate announces a platform:
             with prob p_self  → self-interested proposal
             with prob 1-p_self → constituency-welfare proposal
        4. Build voter utility matrix:
             policy utility   : fitness under each candidate's platform, [0,1]-normalised
             identity utility : Hamming similarity on non-voting bits, already [0,1]
             combined         : (1-beta)*policy_norm + beta*identity
        5. Elect winner via the configured voting method; implement their platform.
        """
        candidate_indices = self._select_candidates(current_fitnesses)
        n_cand    = len(candidate_indices)
        n_nonvote = len(self.non_vote_indices)

        # --- Constituency assignment ---
        nv_all   = self.solutions[:, self.non_vote_indices]                      # (n, L)
        nv_cands = self.solutions[candidate_indices][:, self.non_vote_indices]   # (k, L)
        diff         = nv_all[:, None, :] ^ nv_cands[None, :, :]               # (n, k, L)
        hamming_dist = diff.sum(axis=2)                                          # (n, k)
        assignment   = np.argmin(hamming_dist, axis=1)                          # (n,)

        # Identity utility: normalised Hamming similarity, always in [0, 1]
        identity_util = 1.0 - hamming_dist / n_nonvote                          # (n, k)

        # --- Platform formation ---
        platforms = np.zeros(n_cand, dtype=int)
        for c in range(n_cand):
            if np.random.random() < self.p_self:
                # Trustee: maximise own fitness
                platforms[c] = np.argmax(proposal_fitnesses[candidate_indices[c]])
            else:
                # Delegate: maximise constituency mean welfare
                mask = (assignment == c)
                if mask.sum() > 0:
                    platforms[c] = np.argmax(proposal_fitnesses[mask].mean(axis=0))
                else:
                    # Empty constituency (rare): fall back to self-interest
                    platforms[c] = np.argmax(proposal_fitnesses[candidate_indices[c]])

        # --- Voter utility matrix ---
        # Policy utility: fitness voter i would get under candidate c's platform
        policy_util = proposal_fitnesses[:, platforms]                           # (n, k)

        # Normalise policy utility to [0, 1] for mixing with identity utility
        p_min, p_max = policy_util.min(), policy_util.max()
        if p_max > p_min:
            policy_norm = (policy_util - p_min) / (p_max - p_min)
        else:
            policy_norm = np.full_like(policy_util, 0.5)

        # Combined utility
        combined_util = (1.0 - self.beta) * policy_norm + self.beta * identity_util

        # Approval threshold in combined-utility space:
        #   policy component  → normalised current fitness (status-quo baseline)
        #   identity component → 0.5 (more similar than a random agent)
        c_min, c_max = current_fitnesses.min(), current_fitnesses.max()
        if c_max > c_min:
            current_norm = (current_fitnesses - c_min) / (c_max - c_min)
        else:
            current_norm = np.full_like(current_fitnesses, 0.5)
        threshold = (1.0 - self.beta) * current_norm + self.beta * 0.5

        winner_idx = self._vote(combined_util, threshold)
        return self._decimal_to_binary(platforms[winner_idx], self.vote_size)

    # ------------------------------------------------------------------ #
    # Candidate nomination                                                 #
    # ------------------------------------------------------------------ #

    def _select_candidates(self, current_fitnesses):
        """Select candidate representatives, optionally weighted by fitness."""
        n_cand = min(self.num_candidates, self.num_solutions)
        if self.selection_temperature is None:
            return np.random.choice(self.num_solutions, size=n_cand, replace=False)
        logits = current_fitnesses / self.selection_temperature
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        probs   = weights / np.sum(weights)
        return np.random.choice(self.num_solutions, size=n_cand, replace=False, p=probs)

    # ------------------------------------------------------------------ #
    # Voting methods                                                       #
    # ------------------------------------------------------------------ #

    def _vote(self, utility_matrix, current_fitnesses=None):
        """
        Apply voting method to a utility matrix (voters x options).
        Returns the winning column index.

        current_fitnesses is used by approval as the per-voter approval threshold.
        """
        n_voters, n_options = utility_matrix.shape

        if self.vote_type == "plurality":
            winners = np.argmax(utility_matrix, axis=1)
            return int(np.argmax(np.bincount(winners, minlength=n_options)))

        elif self.vote_type == "approval":
            improved   = utility_matrix > current_fitnesses[:, None]
            any_improved = improved.any(axis=1)
            tied       = utility_matrix == current_fitnesses[:, None]
            approvals  = np.where(any_improved[:, None], improved, tied)
            return int(np.argmax(approvals.sum(axis=0)))

        elif self.vote_type == "borda":
            ranks = np.argsort(np.argsort(utility_matrix, axis=1), axis=1)
            return int(np.argmax(np.sum(ranks, axis=0)))

        elif self.vote_type == "irv":
            rankings  = np.argsort(-utility_matrix, axis=1)
            eliminated = np.zeros(n_options, dtype=bool)
            while True:
                valid       = ~eliminated[rankings]
                first_col   = np.argmax(valid, axis=1)
                first_choice = rankings[np.arange(n_voters), first_col]
                tally       = np.bincount(first_choice, minlength=n_options)
                if tally.max() > n_voters / 2:
                    return int(np.argmax(tally))
                remaining = np.where(~eliminated)[0]
                if len(remaining) == 1:
                    return int(remaining[0])
                tally_f = tally.astype(float)
                tally_f[eliminated] = np.inf
                eliminated[int(np.argmin(tally_f))] = True

        elif self.vote_type == "star":
            scores   = np.sum(utility_matrix, axis=0)
            top_two  = np.argsort(scores)[-2:]
            pa = np.sum(utility_matrix[:, top_two[0]] > utility_matrix[:, top_two[1]])
            pb = np.sum(utility_matrix[:, top_two[1]] > utility_matrix[:, top_two[0]])
            return int(top_two[1]) if pb >= pa else int(top_two[0])

        elif self.vote_type == "minimax":
            pairwise = (utility_matrix[:, :, None] > utility_matrix[:, None, :]).sum(axis=0)
            worst    = pairwise.max(axis=0)
            return int(np.argmin(worst))

        elif self.vote_type == "total_score":
            return int(np.argmax(np.sum(utility_matrix, axis=0)))

        elif self.vote_type == "random_dictator":
            dictator = np.random.randint(n_voters)
            return int(np.argmax(utility_matrix[dictator]))

        elif self.vote_type == "scoring_p035":
            ranks = np.argsort(np.argsort(-utility_matrix, axis=1), axis=1)
            if n_options > 1:
                scores = ((n_options - 1 - ranks) / (n_options - 1)) ** 0.35
            else:
                scores = np.ones_like(ranks, dtype=float)
            return int(np.argmax(scores.sum(axis=0)))

        raise ValueError(f"Unknown vote type: {self.vote_type}")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _calculate_proposal_fitnesses(self, proposal_indices):
        n_proposals = 2 ** self.vote_size
        fitnesses   = np.zeros((self.num_solutions, n_proposals))
        for p in range(n_proposals):
            proposed = self.solutions.copy()
            proposed[:, proposal_indices] = self._decimal_to_binary(p, self.vote_size)
            fitnesses[:, p] = self.nk.calculate_fitness_batch(proposed)
        return fitnesses

    def _update_solutions(self, winner, proposal_indices):
        self.solutions[:, proposal_indices] = winner
        self.solutions    = np.unique(self.solutions, axis=0)
        self.num_solutions = self.solutions.shape[0]

    def _generate_vote_indices(self):
        np.random.shuffle(self.possible_vote_indices)
        return self.possible_vote_indices[:self.vote_size]

    @staticmethod
    def _decimal_to_binary(decimal, length):
        binary = list(map(int, bin(decimal)[2:]))
        return np.array([0] * (length - len(binary)) + binary, dtype=int)

    @staticmethod
    def _binary_to_decimal(binary):
        return sum(b * 2 ** (len(binary) - 1 - i) for i, b in enumerate(binary))

    # ------------------------------------------------------------------ #
    # State access                                                         #
    # ------------------------------------------------------------------ #

    def get_fitnesses(self):
        return self.nk.calculate_fitness_batch(self.solutions)

    def get_mean(self):     return np.mean(self.get_fitnesses())
    def get_variance(self): return np.var(self.get_fitnesses())
    def get_max(self):      return np.max(self.get_fitnesses())
    def get_min(self):      return np.min(self.get_fitnesses())
    def get_solutions(self): return self.solutions
    def get_num_solutions(self): return self.num_solutions
    def get_nk_model(self): return self.nk

    def set_solutions(self, solutions):
        self.solutions     = solutions
        self.num_solutions = self.solutions.shape[0]
