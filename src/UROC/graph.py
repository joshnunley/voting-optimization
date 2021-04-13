import matplotlib.pyplot as plt
import numpy as np

folder = str(input("Folder name: "))

iterations = np.load(folder + '/iterations.npy')  # IMPLEMENT THIS RIGHT NOW
vote_types = np.load(folder + '/vote_types.npy')
mean_history = np.load(folder + '/mean_history.npy')
variance_history = np.load(folder + '/mean_variance.npy')
min_history = np.load(folder + '/mean_min.npy')
max_history = np.load(folder + '/mean_max.npy')

# Calculate mean here
mean_mean_history = np.zeros(shape=(len(vote_types), iterations))
mean_variance_history = np.zeros(shape=(len(vote_types), iterations))
mean_min_history = np.zeros(shape=(len(vote_types), iterations))
mean_max_history = np.zeros(shape=(len(vote_types), iterations))

type_index = 0
for type in vote_types:
    mean_mean_history[type_index, :] = np.mean(mean_history[type_index, :, :], axis=0)
    mean_variance_history[type_index, :] = np.mean(variance_history[type_index, :, :], axis=0)
    mean_min_history[type_index, :] = np.mean(min_history[type_index, :, :], axis=0)
    mean_max_history[type_index, :] = np.mean(max_history[type_index, :, :], axis=0)

    type_index += 1

plt.figure(0)
plt.title("Mean List")
plt.xlabel("Iteration #")
plt.ylabel("Mean")
plot_object = plt.plot(mean_mean_history.transpose())
plt.legend(iter(plot_object), vote_types, loc='upper left')
plt.savefig(folder + "/mean.png")
plt.show()

plt.figure(1)
plt.title("Variance List")
plt.xlabel("Iteration #")
plt.ylabel("Variance")
plot_object = plt.plot(mean_variance_history.transpose())
plt.legend(iter(plot_object), vote_types, loc='upper left')
plt.savefig(folder + "/variance.png")
plt.show()

plt.figure(2)
plt.title("Min List")
plt.xlabel("Iteration #")
plt.ylabel("Min")
plot_object = plt.plot(mean_min_history.T)
plt.legend(iter(plot_object), vote_types, loc='upper left')
plt.savefig(folder + "/min.png")
plt.show()

plt.figure(3)
plt.title("Max List")
plt.xlabel("Iteration #")
plt.ylabel("Max")
plot_object = plt.plot(mean_max_history.T)
plt.legend(iter(plot_object), vote_types, loc='upper left')
plt.savefig(folder + "/max.png")
plt.show()

plt.figure(4)
plt.title("Plurality Voting")
plot_object = plt.plot(mean_history[0, 0:4, :].T)
plt.show()
