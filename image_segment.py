"""
Rotina de segmentacao de imagens.

args:

caminho para imagens, numero de clusters, numero maximo de iteracoes

"""

from kmeans import KMeans
from fcm import FCM
from image_tools import *
import sys
from matplotlib import pyplot as plt

def segment_image(pixels, k, it, tolerance = .001, fuzzy=False):
    """
    Executa algoritmo de clusterizacao sobre conjunto de pixels

    pixels: dataframe com duas colunas contendo pixels

    k: numero de clusters do algoritmo

    it: numero de iteracoes maximas

    tolerance: tolerancia de erro do algoritmo

    fuzzy: se True, executa algoritmo fuzzy
    """

    if fuzzy:
        clustering = FCM(k, pixels.to_numpy())
    else:
        clustering = KMeans(k, pixels.to_numpy())

    clustering.train(it)
    membership = clustering.U.argmax(axis=0)

    return clustering.C, membership


# Carrega pixels 

photo = photo_open(sys.argv[1])
pixels = pick_pixels(photo)

# Clusteriza pixels

centers, membership = segment_image(pixels, int(sys.argv[2]), int(sys.argv[3]))

# Recompoe imagem baseada nos centroides calculados

segmented_photo = coloring(photo, membership, centers)

# Salva imagem
plt.imshow(segmented_photo)
imtag = sys.argv[1].replace(".jpg", "")
plt.savefig(f"{imtag}_segmented_{sys.argv[2]}_{sys.argv[3]}")
