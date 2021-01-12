import numpy as np
from matplotlib import pyplot as plt
import cv2


def load_photo(filename, show=False):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show:
        plt.imshow(img)

    img = img.reshape((-1, 3))

    return img

#TODO: entender recuperação de pixels após FCM

def coloring(photo, labels, centers, rescale=1, show=False):
    n, m = photo.size
    pixels = photo.load()
    for i in range(n):
        for j in range(m):
            # numb = [int(number) for number in centers[labels[i * m + j]]]
            numb = [number for number in centers[labels[i * m + j]].astype(int)]
            # pixels[i, j] = tuple(numb)
            pixels[i, j] = tuple(map(tuple, numb))
    photo = photo.resize((int(photo.size[0] * rescale),
                          int(photo.size[1] * rescale)), Image.ANTIALIAS)

    if show:
        plt.imshow(photo)
    return photo
