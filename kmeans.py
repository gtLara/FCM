import numpy as np
from numpy.random import normal as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

# class definitions

class KMeans():
    def __init__(self, n_clusters, data):
        self.k = n_clusters
        self.X = data
        self.n_samples = len(data)

    def dist(self, a, b):
        return norm(a - b)

    def update_U(self):
        self.U = np.zeros((self.C.shape[0], self.X.shape[0]))

        for s, sample in enumerate(self.X):
            min_dist = np.inf
            for c, centroid in enumerate(self.C):
                current_dist = self.dist(sample, centroid)
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
        centroids = np.mean(self.X[np.where(self.U[0] == 1)], axis=0)
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
                self.cost += self.U[i, j] * self.dist(centroid, sample)

    def show_state(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c="black")
        plt.scatter(self.C[:, 0], self.C[:, 1], c="red")
        plt.show()

    def train(self, it=4, show=False):
        self.U = ri(2, size=(self.k, 240))

        for i in range(it):
            self.update_C()
            self.update_U()
            self.update_cost()
            if show:
                self.show_state()
