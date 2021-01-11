import numpy as np
from numpy.random import normal as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

# class definitions

class k_means():
    def __init__(self, n_clusters, data=None, n_samples=None):
        self.k = n_clusters
        self.data = data
        if n_samples is not None:
            self.n_samples = len(data)
        else:
            self.n_samples = n_samples

    def dist(self, a, b):
        return norm(a - b)

    def update_U(self):
        self.U = np.zeros((self.C.shape[0], self.X.shape[0]))

        for s, sample in enumerate(self.X):
            min_dist = np.inf
            for c, centroid in enumerate(self.C):
                current_dist = dist(sample, centroid)
                if current_dist < min_dist:
                    min_dist = current_dist
                    closest_c = c

            self.U[closest_c, s] = 1


    def update_C(self):

        # if start:
        #     if k is None:
        #         assert("Undefined number of clusters")
        #     return X[ri(len(X), size=k), :]

        cluster_size = sum(self.U[0])
        centroids = np.mean(self.X[np.where(U[0] == 1)], axis=0)
        centroids = centroids/cluster_size

        for i, membership in enumerate(self.U[1:]):
            cluster_size = sum(membership)
            centroid = np.mean(self.X[np.where(membership == 1)], axis=0)
            centroids = np.vstack((centroids, centroid))

        self.C = centroids


    def update_cost(self):
        self.cost = 0
        for i, centroid in enumerate(self.C):
            for j, sample in enumerate(self.X):
                self.cost += U[i, j] * dist(centroid, sample)

# Sample creation
#TODO: implementar rotina de treino e visualização de dados em classe

n_groups = 3
n_samples = 240
group_size = int(n_samples/n_groups)

X = r(loc=0, scale=.33, size=((group_size), 2))

for g in range(n_groups-1):
    new_group = r(loc=g+1, scale=.33, size=((group_size), 2))
    X = np.vstack((X, new_group))

# Initialization

U = ri(2, size=(3, 240))

# Iteration

for it in range(10):
    C = get_C(X, U)
    U = update_U(C, X)
    cost = get_cost(U, C, X)

    plt.scatter(X[:, 0], X[:, 1], c="red")
    plt.scatter(C[:, 0], C[:, 1], c="black")
    plt.show()
