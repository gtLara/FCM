from kmeans import KMeans
import numpy as np
from numpy.random import random as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

class FCM(KMeans):
    """
    Implementa fuzzy k means herdando classe KMeans. A documentacao das
    funcoes herdadas esta presente na superclasse.
    """

    def __init__(self, n_clusters, data, m=2, tolerance=.001):

        """
        Inicializa classe FCM

        n_clusters: numero de clusters (k)

        data: dados que serao ajustados (array numpy)

        tolerance: tolerancia para custo, a partir da qual o treinamento
        se encerra

        m: fator de "fuzzyness" (Jang, p426)

        """
        self.m = m
        KMeans.__init__(self, n_clusters, data, tolerance)


    def update_U(self):
        """
        Atualiza matriz de pertinencia de acordo com equacao 15.5
        """

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

    def train(self, it=4, show=False):
        """
        Rotina de treino como demonstrada no livro texto
        """

        # Inicializa matriz de pertinencia aleatoriamente

        U = np.zeros((self.k, self.N))

        for j in range(self.N):
            membership = r((self.k))
            membership = membership/sum(membership)
            self.U[:, j] = membership

        # Implementa rotina de treino de acordo com o livro texto

        for i in range(it):
            self.update_C()
            self.update_U()
            self.update_cost()

            if self.cost < self.tolerance:
                return self.cost

            if show:
                self.show_state()

        return self.cost
