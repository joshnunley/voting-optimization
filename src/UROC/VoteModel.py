import numpy as np


class VoteModel:
    """
    Collective decision-making model using voting methods on NK landscapes.

    Supports direct democracy (vote on proposals) and representative democracy
    (vote on candidates who then choose proposals).

    Parameters:
        nk_model: NKLandscape instance
        solutions: 2D binary array (num_solutions x N)
        possible_vote_indices: indices allowed to be modified by voting
        vote_size: number of bit positions voted on per round
        vote_type: voting method name
        num_candidates: enables representative democracy with this many
            candidates randomly selected each round. None = direct democracy.
    """

    VOTE_TYPES = [
        "plurality", "approval", "total_score",
        "borda", "irv", "star", "minimax", "random_dictator",
    ]

    def __init__(self, nk_model, solutions, possible_vote_indices=None,
                 vote_size=2, vote_type="plurality", num_candidates=None,
                 selection_temperature=None):
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
        self.vote_size = vote_size
        self.vote_type = vote_type
        self.num_candidates = num_candidates
        self.selection_temperature = selection_temperature

    # ---- Core voting loop ----

    def run(self, iterations=100, until_unique=False):
        for _ in range(iterations):
            self.step()
            if until_unique and self.num_solutions == 1:
                break

    def step(self):
        """Execute a single voting round."""
        proposal_indices = self._generate_vote_indices()
        proposal_fitnesses = self._calculate_proposal_fitnesses(proposal_indices)
        current_fitnesses = self.get_fitnesses()

        if self.num_candidates is None:
            winner_idx = self._vote(proposal_fitnesses, current_fitnesses)
            winner = self._decimal_to_binary(winner_idx, self.vote_size)
        else:
            winner = self._representative_step(
                proposal_fitnesses, current_fitnesses
            )

        self._update_solutions(winner, proposal_indices)

    def _select_candidates(self, current_fitnesses):
        """Select candidate representatives, optionally weighted by fitness."""
        n_cand = min(self.num_candidates, self.num_solutions)
        if self.selection_temperature is None:
            return np.random.choice(self.num_solutions, size=n_cand, replace=False)
        # Softmax-weighted selection
        logits = current_fitnesses / self.selection_temperature
        logits = logits - np.max(logits)  # numerical stability
        weights = np.exp(logits)
        probs = weights / np.sum(weights)
        return np.random.choice(
            self.num_solutions, size=n_cand, replace=False, p=probs
        )

    def _representative_step(self, proposal_fitnesses, current_fitnesses):
        """Representative democracy: vote on candidates, winner picks proposal."""
        candidate_indices = self._select_candidates(current_fitnesses)
        # Each candidate's preferred proposal
        preferred = np.argmax(proposal_fitnesses[candidate_indices], axis=1)
        # Voter x candidate utility: fitness voter i gets under candidate c's choice
        candidate_utilities = proposal_fitnesses[:, preferred]
        # Vote on candidates
        winner_idx = self._vote(candidate_utilities, current_fitnesses)
        # Return winning candidate's preferred proposal as binary
        return self._decimal_to_binary(preferred[winner_idx], self.vote_size)

    # ---- Voting methods ----

    def _vote(self, utility_matrix, current_fitnesses=None):
        """
        Apply voting method to a utility matrix (voters x options).
        Returns the winning column index.
        """
        n_voters, n_options = utility_matrix.shape

        if self.vote_type == "plurality":
            tally = np.zeros(n_options)
            for i in range(n_voters):
                tally[np.argmax(utility_matrix[i])] += 1
            return int(np.argmax(tally))

        elif self.vote_type == "approval":
            tally = np.zeros(n_options)
            for i in range(n_voters):
                improved = utility_matrix[i] > current_fitnesses[i]
                if np.any(improved):
                    tally[improved] += 1
                else:
                    tied = utility_matrix[i] == current_fitnesses[i]
                    tally[tied] += 1
            return int(np.argmax(tally))

        elif self.vote_type == "borda":
            ranks = np.argsort(np.argsort(utility_matrix, axis=1), axis=1)
            return int(np.argmax(np.sum(ranks, axis=0)))

        elif self.vote_type == "irv":
            rankings = np.argsort(-utility_matrix, axis=1)
            eliminated = set()
            while True:
                tally = np.zeros(n_options)
                for i in range(n_voters):
                    for r in range(n_options):
                        c = rankings[i, r]
                        if c not in eliminated:
                            tally[c] += 1
                            break
                if np.max(tally) > n_voters / 2:
                    return int(np.argmax(tally))
                remaining = [j for j in range(n_options) if j not in eliminated]
                if len(remaining) == 1:
                    return remaining[0]
                min_votes, min_cand = float("inf"), -1
                for j in range(n_options):
                    if j not in eliminated and tally[j] < min_votes:
                        min_votes, min_cand = tally[j], j
                eliminated.add(min_cand)

        elif self.vote_type == "star":
            scores = np.sum(utility_matrix, axis=0)
            top_two = np.argsort(scores)[-2:]
            pa = np.sum(utility_matrix[:, top_two[0]] > utility_matrix[:, top_two[1]])
            pb = np.sum(utility_matrix[:, top_two[1]] > utility_matrix[:, top_two[0]])
            return int(top_two[1]) if pb >= pa else int(top_two[0])

        elif self.vote_type == "minimax":
            pairwise = np.zeros((n_options, n_options))
            for a in range(n_options):
                for b in range(n_options):
                    if a != b:
                        pairwise[a, b] = np.sum(
                            utility_matrix[:, a] > utility_matrix[:, b]
                        )
            worst = np.array([
                max(pairwise[b, a] for b in range(n_options) if b != a)
                for a in range(n_options)
            ])
            return int(np.argmin(worst))

        elif self.vote_type == "total_score":
            return int(np.argmax(np.sum(utility_matrix, axis=0)))

        elif self.vote_type == "random_dictator":
            dictator = np.random.randint(n_voters)
            return int(np.argmax(utility_matrix[dictator]))

        raise ValueError(f"Unknown vote type: {self.vote_type}")

    # ---- Helpers ----

    def _calculate_proposal_fitnesses(self, proposal_indices):
        n_proposals = 2 ** self.vote_size
        fitnesses = np.zeros((self.num_solutions, n_proposals))
        for i, solution in enumerate(self.solutions):
            for p in range(n_proposals):
                proposed = np.copy(solution)
                np.put(proposed, proposal_indices,
                       self._decimal_to_binary(p, self.vote_size))
                fitnesses[i, p] = self.nk.calculate_fitness(proposed)
        return fitnesses

    def _update_solutions(self, winner, proposal_indices):
        for i in range(self.num_solutions):
            np.put(self.solutions[i], proposal_indices, winner)
        self.solutions = np.unique(self.solutions, axis=0)
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

    # ---- State access ----

    def get_fitnesses(self):
        return np.array([
            self.nk.calculate_fitness(self.solutions[i])
            for i in range(self.num_solutions)
        ])

    def get_mean(self):
        return np.mean(self.get_fitnesses())

    def get_variance(self):
        return np.var(self.get_fitnesses())

    def get_max(self):
        return np.max(self.get_fitnesses())

    def get_min(self):
        return np.min(self.get_fitnesses())

    def get_solutions(self):
        return self.solutions

    def set_solutions(self, solutions):
        self.solutions = solutions
        self.num_solutions = self.solutions.shape[0]

    def get_num_solutions(self):
        return self.num_solutions

    def get_nk_model(self):
        return self.nk
