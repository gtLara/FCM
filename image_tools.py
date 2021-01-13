import numpy as np
from PIL import Image
import pandas as pd

def photo_open(filename, rescale = 1):
    photo = Image.open(filename)
    photo = photo.convert('RGB')
    photo = photo.resize((int(photo.size[0] / rescale),
                          int(photo.size[1] / rescale)), Image.ANTIALIAS)
    return photo

def pick_pixels(photo):
    n , m = photo.size
    ibagem = []
    pixels = photo.load()
    for i in range(n):
        for j in range(m):
            ibagem.append(list(pixels[i,j]) )
    return pd.DataFrame(ibagem)

def coloring(photo, labels, centers, rescale=1):
    n, m = photo.size
    pixels = photo.load()
    for i in range(n):
        for j in range(m):
            numb = [int(number) for number in centers[labels[i*m + j]] ] 
            pixels[i,j]= tuple(numb)
    photo = photo.resize( (int( photo.size[0]*rescale), int(photo.size[1]*rescale)), Image.ANTIALIAS)
    return photo

#def load_photo(filename, reshape=True, show=False):

#    img = cv2.imread(filename)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    print(img.shape)

#    if show:
#        plt.imshow(img)
#        plt.show()

#    if reshape:
#        img = img.reshape((-1, 3))

#    return img

##TODO: entender recuperação de pixels após FCM

#def recolor(pixels, h, w, membership, centers, show=False):
#    pass

#def coloring(filename, labels, centers, rescale=1, show=False):

#    pixels = load_photo(filename, reshape=False, show=True)
#    n, m = pixels.shape[0], pixels.shape[1]
#    print(n)
#    print(m)

#    for i in range(n):
#        for j in range(m): #m
#            # numb = [int(number) for number in centers[labels[i * m + j]]]

#            # m: largura
#            # n: altura

#            numb = [number for number in centers[labels[i * m + j]].astype(int)]
#            # pixels[i, j] = tuple(numb)
#            pixels[i, j] = tuple(map(tuple, numb))

#    return pixels
