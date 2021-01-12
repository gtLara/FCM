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
        self.N = self.X.shape[0]
        self.n = self.X.shape[1]
        self.U = np.zeros((self.k, self.N))
        self.C = np.zeros((self.k, self.n))

    def dist(self, a, b):
        return norm(a - b)

    def update_U(self):

        self.U = np.zeros((self.k, self.N))

        for s, sample in enumerate(self.X):
            min_dist = np.inf
            for c, centroid in enumerate(self.C):
                current_dist = self.dist(sample, centroid)
                if current_dist < min_dist:
                    min_dist = current_dist
                    closest_c = c

            self.U[closest_c, s] = 1

    def update_C(self):

        for i in range(self.k):
            centroid = np.zeros((self.n))
            cluster_size = sum(self.U[i])
            if cluster_size == 0:
                centroid = self.C[i]
            else:
                for j in range(self.N):
                    centroid += self.U[i,j]*self.X[j]

                centroid /= cluster_size
                print(centroid)

            self.C[i] = centroid
        print()

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

        for j in range(self.N):
            membership = np.zeros((self.k))
            membership[ri((self.k))] = 1
            self.U[:, j] = membership

        for i in range(it):
            self.update_C()
            self.update_U()
            self.update_cost()
            if show:
                self.show_state()
