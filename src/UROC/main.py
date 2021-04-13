import numpy as np
from VoteModel import VoteModel
from NKLandscape import NKLandscape
import os


folder = str(input("Folder name: "))
os.mkdir(folder)


seed = 0
np.random.seed(seed)


def generate_solutions(num, n, seed):
    np.random.seed(seed)
    return np.random.randint(2, size=(num, n))


n = 50
k = 5  # 0, 1, 5, 10, 15, 20
voting_portion = 0.5  # 0.25, 0.5, 0.75
num_solutions = 100
nk = NKLandscape(n, k)

initial_solutions = np.zeros(shape=(num_solutions, n)).astype(int)
non_voting_bits = generate_solutions(num_solutions, int((1 - voting_portion) * n), seed)
voting_bits = generate_solutions(1, int(voting_portion * n), seed)

indices = np.arange(n)
np.random.shuffle(indices)
possible_vote_indices = indices[: int(voting_portion * n)]
not_possible_vote_indices = [x for x in indices if x not in possible_vote_indices]
initial_solutions[:, possible_vote_indices] = voting_bits
initial_solutions[:, not_possible_vote_indices] = non_voting_bits


iterations = 100
runs = 50
vote_types = [
    "plurality",
    "approval",
    "normalized_score",
    "total_score",
    "marginal_score",
]
mean_history = np.zeros(shape=(len(vote_types), runs, iterations))
variance_history = np.zeros(shape=(len(vote_types), runs, iterations))
min_history = np.zeros(shape=(len(vote_types), runs, iterations))
max_history = np.zeros(shape=(len(vote_types), runs, iterations))

type_index = 0

for voting_type in vote_types:
    vote = VoteModel(
        nk,
        solutions=initial_solutions,
        possible_vote_indices=possible_vote_indices,
        vote_size=2,
        vote_type=voting_type,
    )

    for k in range(runs):
        for i in range(iterations):
            vote.run(iterations=1, until_unique=False, verbose=False)
            mean_history[type_index, k, i] = vote.get_mean()
            variance_history[type_index, k, i] = vote.get_variance()
            min_history[type_index, k, i] = vote.get_min()
            max_history[type_index, k, i] = vote.get_max()
            print(voting_type, i)
    type_index += 1

np.save((folder + "/mean_history"), mean_history)
np.save((folder + "/mean_variance"), variance_history)
np.save((folder + "/mean_max"), max_history)
np.save((folder + "/mean_min"), min_history)
np.save((folder + "/vote_types"), vote_types)
np.save((folder + "/solutions"), initial_solutions)
np.save((folder + "/iterations"), iterations)
np.save((folder + "/n"), n)
np.save((folder + "/k"), k)
np.save((folder + "/voting_portion"), voting_portion)
np.save((folder + "/runs"), runs)
np.save((folder + "/iterations"), iterations)
np.save((folder + "/fitness"), nk.get_fitness_mapping())

print("Saving Process Complete")
