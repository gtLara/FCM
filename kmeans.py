import numpy as np
from numpy.random import normal as r
from numpy.random import randint as ri
from matplotlib import pyplot as plt
from numpy.linalg import norm

class KMeans():
    """
    Definicao de classe para algoritmo kmeans classico
    """
    def __init__(self, n_clusters, data, tolerance=.001):
        """
        Inicializa classe KMeans

        n_clusters: numero de clusters (k)

        data: dados que serao ajustados (array numpy)

        tolerance: tolerancia para custo, a partir da qual o treinamento
        se encerra

        """

        self.k = n_clusters
        self.X = data
        self.N = self.X.shape[0]
        self.n = self.X.shape[1]
        self.U = np.zeros((self.k, self.N))
        self.C = np.zeros((self.k, self.n))
        self.tolerance = tolerance
        self.loss_tracker = []

    def dist(self, a, b):
        """
        Calcula norma euclidiana ao quadrado
        """
        return norm(a - b)**2

    def update_U(self):
        """
        Atualiza matriz de pertinencia de acordo com equacao 15.3
        """
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
        """
        Atualiza matriz C de acordo com equacao 15.8
        """

        for i in range(self.k):
            centroid = np.zeros((self.n))
            cluster_size = sum(self.U[i])
            if cluster_size == 0:
                centroid = self.C[i]
            else:
                for j in range(self.N):
                    centroid += self.U[i,j]*self.X[j]

                centroid /= cluster_size

            self.C[i] = centroid

    def update_cost(self):
        """
        Atualiza custo de acordo com equacao 15.1
        """

        self.cost = 0
        for i, centroid in enumerate(self.C):
            for j, sample in enumerate(self.X):
                self.cost += self.U[i, j] * self.dist(centroid, sample)

        self.loss_tracker.append(self.cost)

    def show_loss(self, it):
        plt.plot(self.loss_tracker, color="red")
        plt.title(f"Visualização de Loss para {it} iterações")


    def show_state(self):
        """
        Visualiza estado da clusterizacao
        """

        plt.scatter(self.X[:, 0], self.X[:, 1], c="black")
        plt.scatter(self.C[:, 0], self.C[:, 1], c="red")
        plt.show()

    def train(self, it=4, show=False):
        """
        Rotina de treino como demonstrada no livro texto
        """

        # Inicializa matriz de pertinencia aleatoriamente

        for j in range(self.N):
            membership = np.zeros((self.k))
            membership[ri((self.k))] = 1
            self.U[:, j] = membership

        # Loop de treino

        for i in range(it):
            self.update_C()
            self.update_U()
            self.update_cost()

            if self.cost < self.tolerance:
                return
            if show:
                self.show_state()

        return self.cost
