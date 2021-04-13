import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class VoteModel:
    """
    Summary:
        Optimization algorithm using voting methods

    Required Parameters:
        nk_model is an instance of the NKLandscape class (though, as an optimization method,
        this parameter could take any class instance which has binary array solutions of fixed size N and
        a calculate_fitness method)

        solutions is a 2d binary numpy array of shape (num_solutions, N) where N is the size of solutions
        to the provided nk model

    Optional Parameters:
        possible_vote_indices is the region of the solution that is allowed to be modified.
        If this parameter is not provided, the entire solution is allowed to be modified.

        vote_size determines the number of bit positions to vote on in each voting round

        vote_type determines the voting model used to decide which proposal to implement


    To do:
        -Implement Plurality with Runoff

        -Implement Borda Rule

        -Implement Hare Rule

        -Implement Coombs Rule

    """

    def __init__(
        self,
        nk_model=None,
        solutions=None,
        possible_vote_indices=None,
        vote_size=2,
        vote_type="plurality",
    ):
        if nk_model is None:
            raise ValueError("An NKLandscape model must be provided")
        self.nk = nk_model

        if solutions is None:
            raise ValueError("A set of voting solutions must be provided")
        self.solutions = np.copy(solutions)

        if possible_vote_indices is None:
            self.possible_vote_indices = np.arange(self.nk.N)
        else:
            self.possible_vote_indices = possible_vote_indices

        self.num_solutions = solutions.shape[0]

        self.vote_size = vote_size
        self.vote_type = vote_type

        self.mean_list = np.zeros(shape=(50, 2))

    def _calculate_proposal_fitnesses(self, proposal_indices):
        proposal_fitnesses = np.zeros((self.num_solutions, 2 ** self.vote_size))
        for i, solution in enumerate(self.solutions):
            for proposal in range(2 ** self.vote_size):
                binary_proposal = self._decimal_to_binary(proposal, self.vote_size)

                proposed_solution = np.copy(solution)
                np.put(proposed_solution, proposal_indices, binary_proposal)

                proposal_fitnesses[i, proposal] = self.nk.calculate_fitness(
                    proposed_solution
                )

        return proposal_fitnesses

    def _decimal_to_binary(self, decimal, length):
        binary = list(map(int, bin(decimal)[2:]))
        pad = length - len(binary)
        binary = [0] * pad + binary

        return np.asarray(binary, int)

    def _binary_to_decimal(self, binary):
        decimal = 0
        for i in range(len(binary)):
            decimal += binary[len(binary) - i - 1] * 2 ** i

        return decimal

    def _determine_winner(self, proposal_fitnesses):
        if self.vote_type == "plurality":
            tally = np.zeros(2 ** self.vote_size)
            for i in range(self.num_solutions):
                vote = np.argwhere(
                    proposal_fitnesses[i] == np.max(proposal_fitnesses[i])
                )[0]
                tally[vote[0]] += 1
            print(tally)
            decimal_winner = np.argwhere(tally == np.max(tally))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)

        elif self.vote_type == "approval":
            tally = np.zeros(2 ** self.vote_size)
            current_fitnesses = self.get_fitnesses()
            for i in range(self.num_solutions):
                votes = np.argwhere(proposal_fitnesses[i] > current_fitnesses[i]).T

                if len(votes[0]) > 0:
                    tally[votes[0]] += 1
                else:
                    self_vote = np.argwhere(
                        proposal_fitnesses[i] == current_fitnesses[i]
                    ).T
                    tally[self_vote[0]] += 1

            decimal_winner = np.argwhere(tally == np.max(tally))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)

        elif self.vote_type == "ranked":  # This is where new code started
            current_fitnesses = self.get_fitnesses()[np.newaxis].T
            marginal_scores = proposal_fitnesses - np.tile(
                current_fitnesses, 2 ** self.vote_size
            )
            scores = np.sort(marginal_scores, axis=1)
            # Implement Borda Rule here

            decimal_winner = np.argwhere(scores == np.max(scores))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)

        # A version of score that is most similar to actual score voting (actually more like cumulative voting).
        # Each solution has the same amount of score (1) to alot to the proposals
        elif self.vote_type == "normalized_score":
            minimum_proposal_fitnesses = np.min(proposal_fitnesses, axis=1)[
                np.newaxis
            ].T
            minimum_proposal_fitnesses = np.tile(
                minimum_proposal_fitnesses, 2 ** self.vote_size
            )
            positive_proposal_fitnesses = (
                proposal_fitnesses - minimum_proposal_fitnesses
            )

            normalizing_factors = np.sum(positive_proposal_fitnesses, axis=1)[
                np.newaxis
            ].T
            normalizing_factors = np.tile(normalizing_factors, 2 ** self.vote_size)
            normalized_scores = np.divide(
                positive_proposal_fitnesses, normalizing_factors
            )

            scores = np.sum(normalized_scores, axis=0)
            decimal_winner = np.argwhere(scores == np.max(scores))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)
        # These assume voters know how their utility differs from another's (in the case of nk
        # landscapes this is obviously true)
        elif self.vote_type == "total_score":
            scores = np.sum(proposal_fitnesses, axis=0)
            decimal_winner = np.argwhere(scores == np.max(scores))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)
        # I think this should be the optimal setting for optimization
        elif self.vote_type == "marginal_score":
            current_fitnesses = self.get_fitnesses()[np.newaxis].T
            marginal_scores = proposal_fitnesses - np.tile(
                current_fitnesses, 2 ** self.vote_size
            )
            scores = np.sum(marginal_scores, axis=0)
            print(scores)
            decimal_winner = np.argwhere(scores == np.max(scores))[0][0]
            binary_winner = self._decimal_to_binary(decimal_winner, self.vote_size)

        return binary_winner

    def _update_solutions(self, winner, proposal_indices):
        for i in range(self.num_solutions):
            np.put(self.solutions[i], proposal_indices, winner)

        self.solutions = np.unique(self.solutions, axis=0)
        self.num_solutions = self.solutions.shape[0]

    def _generate_vote_indicies(self):
        np.random.shuffle(self.possible_vote_indices)

        return self.possible_vote_indices[0 : self.vote_size]

    def print_distribution(self, i, verbose):
        if verbose:
            fitnesses = self.get_fitnesses()
            print(
                "\n vote iterations:",
                i + 1,
                "num_solutions:",
                self.get_num_solutions(),
                "\n mean:",
                np.mean(fitnesses),
                "variance:",
                np.var(fitnesses),
                "\n min:",
                np.min(fitnesses),
                "max:",
                np.max(fitnesses),
            )
            sns.distplot(fitnesses)
            plt.show()

    def get_mean(self):
        fitnesses = self.get_fitnesses()
        return np.mean(fitnesses)

    def get_variance(self):
        fitnesses = self.get_fitnesses()
        return np.var(fitnesses)

    def get_max(self):
        fitness = self.get_fitnesses()
        return np.max(fitness)

    def get_min(self):
        fitness = self.get_fitnesses()
        return np.min(fitness)

    def run(self, iterations=100, until_unique=False, verbose=False):
        for i in range(iterations):
            proposal_indices = self._generate_vote_indicies()

            proposal_fitnesses = self._calculate_proposal_fitnesses(proposal_indices)

            winner = self._determine_winner(proposal_fitnesses)

            self._update_solutions(winner, proposal_indices)

            self.print_distribution(i, verbose)

            self.mean_list[i] = [i, self.get_mean()]

            if until_unique and (self.num_solutions == 1):
                break

    def get_solutions(self):
        return self.solutions

    def set_solutions(self, solutions):
        self.solutions = solutions
        self.num_solutions = self.solutions.shape[0]

    def get_num_solutions(self):
        return self.num_solutions

    def get_fitnesses(self):
        fitnesses = np.zeros(self.num_solutions)

        for i in range(self.num_solutions):
            fitnesses[i] = self.nk.calculate_fitness(self.solutions[i])

        return fitnesses

    def get_nk_model(self):
        return self.nk
