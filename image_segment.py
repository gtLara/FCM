from kmeans import KMeans
from fcm import FCM
from image_tools import *
import sys
from matplotlib import pyplot as plt

def segment_image(pixels, k, it, fuzzy=True):

    if fuzzy:
        clustering = FCM(k, pixels.to_numpy())
    else:
        clustering = KMeans(k, pixels.to_numpy())

    clustering.train(it)
    membership = clustering.U.argmax(axis=0)

    return clustering.C, membership

photo = photo_open(sys.argv[1])
pixels = pick_pixels(photo)

centers, membership = segment_image(pixels, int(sys.argv[2]), int(sys.argv[3]))
segmented_photo = coloring(photo, membership, centers)

plt.imshow(segmented_photo)
plt.show()
