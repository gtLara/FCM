from kmeans import KMeans
import numpy as np
from numpy.random import random as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

class FCM(KMeans):

    def __init__(self, n_clusters, data, m=2):
        self.m = m
        KMeans.__init__(self, n_clusters, data)

    def update_U(self):

        self.U = np.zeros((self.k, self.N))

        for i in range(self.k):
            for j in range(self.N):
                distance_sum = 0
                for k in range(self.k):
                    internal_norm = norm(self.C[i] - self.X[j])
                    external_norm = norm(self.C[k] - self.X[j])
                    distance_sum += (internal_norm/external_norm) ** (2/(self.m -1))

                self.U[i, j] = 1/distance_sum

        for r in range(self.N):
            self.U[:, r] = self.U[:, r]/sum(self.U[:, r])

    def update_cost(self):
        pass

    def train(self, it=4, show=False):

        U = np.zeros((self.k, self.N))

        for j in range(self.N):
            membership = r((self.k))
            membership = membership/sum(membership)
            self.U[:, j] = membership


        for i in range(it):
            self.update_C()
            self.update_U()
            self.update_cost()
            if show:
                self.show_state()
