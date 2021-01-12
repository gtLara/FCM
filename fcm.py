from kmeans import KMeans
import numpy as np
from numpy.random import normal as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

class FCM(KMeans):
    def __init__(self, n_clusters, data, m):
        self.m = m
        KMeans.__init__(self, n_clusters, data)

    def dist(self, a, b):
        return(norm(a-b)**(2/self.m - 1))

    def inv_dist(self, a, b):
        return((1/norm((a-b)))**(2/(self.m - 1)))

    def update_U(self):

        self.U = np.zeros((self.k, self.N))
        for i in range(self.k):
            for j in range(self.N):
                external_distance = 0
                internal_distance = self.dist(self.X[j], self.C[i])
                for k in range(self.k):
                    external_distance += self.inv_dist(self.X[j], self.C[k])

            self.U[i, k] = 1/(internal_distance * external_distance)

    def update_cost(self):
        pass
