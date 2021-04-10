import numpy as np
from VoteModel import VoteModel
from NKLandscape import NKLandscape
from os import mkdir

folder = str(input("Folder name: "))
mkdir(folder)

seed = 0
np.random.seed(seed)

# 50 and 12, 30 and 6 (in that range)
n = 3
k = 2
nk = NKLandscape(n, k)
solutions = nk.generate_solutions(5, seed)
print(solutions)

# One above .5, one below, and one at
voting_portion = .5
indices = np.arange(nk.N)
np.random.shuffle(indices)
possible_vote_indices = indices[:int(voting_portion * nk.N)]

initial_solutions = np.copy(solutions)

# Somewhere between 50 and 100 for iterations AND runs
iterations = 4  # 100
runs = 2  # 10 up to 50? Depending on output
vote_types = ['plurality', 'approval', 'normalized_score', 'total_score', 'marginal_score', 'ranked']
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
        vote_type=voting_type
    )

    for k in range(runs):
        for i in range(iterations):
            vote.run(iterations=1, until_unique=False, verbose=False)
            mean_history[type_index, k, i] = vote.get_mean()
            variance_history[type_index, k, i] = vote.get_variance()
            min_history[type_index, k, i] = vote.get_min()
            max_history[type_index, k, i] = vote.get_max()
    type_index += 1

np.save((folder + '/mean_history'), mean_history)
np.save((folder + '/mean_variance'), variance_history)
np.save((folder + '/mean_max'), max_history)
np.save((folder + '/mean_min'), min_history)
np.save((folder + '/vote_types'), vote_types)
np.save((folder + '/solutions'), solutions)
np.save((folder + '/iterations'), iterations)
np.save((folder + '/n'), n)
np.save((folder + '/k'), k)
np.save((folder + '/voting_portion'), voting_portion)
np.save((folder + '/runs'), runs)
np.save((folder + '/iterations'), iterations)
np.save((folder + '/fitness'), nk.get_fitness_mapping())

print("Saving Process Complete")
