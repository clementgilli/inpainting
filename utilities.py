import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d

def gradient(ar):
    grad_i, grad_j = np.gradient(ar)
    return np.array([[(grad_i[i, j], grad_j[i, j]) for j in range(grad_i.shape[1])] for i in range(grad_j.shape[0])])

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