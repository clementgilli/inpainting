import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d
from sklearn.neighbors import BallTree
import numba

@numba.jit(nopython=True)
def masked_dist(patch1, patch2, mask):
    dist = np.linalg.norm((patch1 - patch2) * mask)
    return dist

def get_max_dict(dict, value=False):
    if value: # return key and value
        return max(dict.items(), key=lambda k: k[1])
    else: # return only key
        return max(dict,key=dict.get)
    
def masque_carre(c1,c2,imgsize):
    masque = np.zeros((imgsize[0],imgsize[1]))
    masque[c1[0]:c2[0]+1,c1[1]:c2[1]+1] = 1
    return masque

def masque_circulaire(c,r,imgsize):
    masque = np.zeros((imgsize[0],imgsize[1]))
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            if (i-c[0])**2+(j-c[1])**2 <= r**2:
                masque[i,j] = 1
    return masque

def orthogonal_vector(v):
    return np.array([-v[1], v[0]])

def below_line(x,y, a,b, c,d):
    if a == c:
        return x < a
    else:
        return y - ((d-b)/(c-a)*(x-a)+b) > 0 