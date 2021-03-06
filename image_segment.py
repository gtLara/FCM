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

    return clustering.C, membership, clustering

# Carrega parametros

ppath = sys.argv[1]
k = int(sys.argv[2])
it = int(sys.argv[3])

fuzzy = (sys.argv[4] == "fuzzy")

# Carrega pixels

photo = photo_open(ppath)
pixels = pick_pixels(photo)

# Clusteriza pixels

centers, membership, clusters = segment_image(pixels, k, it, fuzzy=fuzzy)

# Recompoe imagem baseada nos centroides calculados

segmented_photo = coloring(photo, membership, centers)

# Salva imagem
plt.imshow(segmented_photo)
imtag = sys.argv[1].split("/")[-1].replace(".jpg", "")

if fuzzy:
    means = "fcm"
else:
    means = "kmeans"

plt.savefig(f"images/{means}/{k}/{imtag}_{it}")
plt.close()
clusters.show_loss(it)
plt.savefig(f"images/{means}/{k}/{imtag}_{it}_loss")
plt.close()
clusters.show_state()
plt.savefig(f"images/{means}/{k}/{imtag}_{it}_state")
