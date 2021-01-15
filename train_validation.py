from scipy.io import loadmat
import sys
from fcm import FCM
import numpy as np
from numpy.random import normal as r
from numpy.random import randint as ri

synthetic = False

# Synthetic data creation
if synthetic:

    n_groups = 5
    n_samples = 240
    group_size = int(n_samples/n_groups)

    X = r(loc=0, scale=.33, size=((group_size), 2))

    for g in range(n_groups-1):
        new_group = r(loc=g+1, scale=.33, size=((group_size), 2))
        X = np.vstack((X, new_group))

    kmeans = FCM(n_groups, X)
    kmeans.train(5, True)

else:

    X = np.array(loadmat("../fcm_dataset.mat")["x"])

    kmeans = FCM(4, X, 2)
    kmeans.train(int(sys.argv[1]), True)

