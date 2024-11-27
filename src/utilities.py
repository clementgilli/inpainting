import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d
from sklearn.neighbors import BallTree
import numba
from skimage.color import rgb2gray
import imageio
import os
from time import time
    
def masque_carre(c1,c2,imgsize,color=False):
    if color:
        masque = np.zeros((imgsize[0],imgsize[1],3))
        masque[c1[0]:c2[0]+1,c1[1]:c2[1]+1] = [1,1,1]
    else:
        masque = np.zeros((imgsize[0],imgsize[1]))
        masque[c1[0]:c2[0]+1,c1[1]:c2[1]+1] = 1
    return masque

def masque_circulaire(c,r,imgsize,color=False):
    if color:
        masque = np.zeros((imgsize[0],imgsize[1],3))
    else:
        masque = np.zeros((imgsize[0],imgsize[1]))
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            if (i-c[0])**2+(j-c[1])**2 <= r**2:
                if color:
                    masque[i,j] = [1,1,1]
                else:
                    masque[i,j] = 1
    return masque

@numba.jit(nopython=True)
def search_zone_compiled(height,width,size,masque,size_search):
        minx,maxx,miny,maxy = width+1,-1,height+1,-1
        for i in range(height):
            for j in range(width):
                if masque[i,j] == 1:
                    if i < miny:
                        miny = i
                    elif i > maxy:
                        maxy = i 
                    if j < minx:
                        minx = j
                    elif j > maxx:
                        maxx = j
        if miny-size_search < 0 + size:
            miny = size+size_search
        if minx-size_search < 0 + size:
            minx = size+size_search
        if maxy+size_search >= height-size:
            maxy = height-size - size_search
        if maxx+size_search >= width-size:
            maxx = width-size - size_search
        
        return (miny-size_search,maxy+size_search),(minx-size_search,maxx+size_search)

@numba.jit(nopython=True)
def masked_dist(patch1, patch2, mask):
    dist = np.linalg.norm((patch1 - patch2) * mask)
    return dist #* hellinger_distance(patch1,patch2)

@numba.jit(nopython=True)
def hellinger_distance(patch1, patch2, bins=16):
    patch1 = patch1.reshape((9, 9, 3))
    patch2 = patch2.reshape((9, 9, 3))
    
    histo1_r = np.histogram(patch1[:, :, 0], bins=bins, range=(0, 256))[0]/81
    histo1_g = np.histogram(patch1[:, :, 1], bins=bins, range=(0, 256))[0]/81
    histo1_b = np.histogram(patch1[:, :, 2], bins=bins, range=(0, 256))[0]/81
    
    histo2_r = np.histogram(patch2[:, :, 0], bins=bins, range=(0, 256))[0]/81
    histo2_g = np.histogram(patch2[:, :, 1], bins=bins, range=(0, 256))[0]/81
    histo2_b = np.histogram(patch2[:, :, 2], bins=bins, range=(0, 256))[0]/81
    
    hellinger_r = np.sqrt(1 - np.sum(np.sqrt(histo1_r * histo2_r)))
    hellinger_g = np.sqrt(1 - np.sum(np.sqrt(histo1_g * histo2_g)))
    hellinger_b = np.sqrt(1 - np.sum(np.sqrt(histo1_b * histo2_b)))
    
    distance = (hellinger_r + hellinger_g + hellinger_b) / 3
    return distance

@numba.jit(nopython=True)
def neighbours(i,j):
    return [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]

@numba.jit(nopython=True)
def outlines_target_compiled(height,width,masque):
        outlines = []
        for i in range(height):
            for j in range(width):
                if masque[i][j] == 1:
                    for n in neighbours(i,j):
                        if masque[n[0]][n[1]] == 0 and n not in outlines:
                            outlines.append(n)
        return outlines

@numba.jit(nopython=True)
def orthogonal_vector(v):
    return np.array([-v[1], v[0]])

@numba.jit(nopython=True)
def below_line(x,y, a,b, c,d):
    if a - c == 0: #pour eviter la division par 0 : cas d'une droite verticale
        return x < a
    else:
        return y - ((d-b)/(c-a)*(x-a)+b) > 0 

def mean_valid_gradient_compiled(i, j, grad_y, grad_x, height, width):
        #Compute the mean gradients of the valid (non-NaN) neighbors around a given point (i, j) in a 2D arrayfor both x and y directions.
        
    valid_gradients_y = []
    valid_gradients_x = []

        # Check the neighbors' gradients; considering the Moore neighborhood (8 surrounding cells)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if di == 0 and dj == 0:
                continue  # Skip the center point itself
            if 0 <= ni < height and 0 <= nj < width:
                    # Append valid (non-NaN) gradients from each dimension
                if not np.isnan(grad_y[ni, nj]):
                    valid_gradients_y.append(grad_y[ni, nj])
                if not np.isnan(grad_x[ni, nj]):
                    valid_gradients_x.append(grad_x[ni, nj])

        # Calculate the mean of valid gradients for each axis

    if valid_gradients_y == []:
        print("bug")
        valid_gradients_y = [0] #à changer
    if valid_gradients_x == []:
        valid_gradients_x = [0] #à changer
        print("bug")
        #print(i,j)
        #plt.imshow(grad_y)
        #plt.show()
        #plt.imshow(grad_x)
        #plt.show()

        #raise ValueError('no valid gradients found')
    
    mean_gradient_y = np.mean(np.array(valid_gradients_y))
    mean_gradient_x = np.mean(np.array(valid_gradients_x))

    return (mean_gradient_y, mean_gradient_x)

@numba.jit(nopython=True)
def compute_normal_compiled(coord, zone, height, width):
    i,j = coord
    if zone[i,j] != 1:
        raise ValueError('trying to calculate normal vector not in frontier')
        
    border_neighbors = []
    target_neighbors = (-1,-1)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if di == 0 and dj == 0:
                continue  # Skip the center point itself
            if 0 <= ni < height and 0 <= nj < width:
                if zone[ni,nj] == 1:
                    border_neighbors.append((ni, nj))
                elif zone[ni,nj] == 0:
                    target_neighbors = (ni,nj)

    if len(border_neighbors) < 2:
        return (0,0)
        #print("bug à",coord)
        #raise ValueError('no target neighbors found')

    border_neighbors=sorted(border_neighbors)
    a,b,c,d = border_neighbors[0][0],border_neighbors[0][1],border_neighbors[-1][0],border_neighbors[-1][1]
    x,y = target_neighbors

    tengeante_x,tengeante_y = a-c,b-d
    norme = (tengeante_x**2+tengeante_y**2)**0.5
        
    if below_line(x,y, a,b, c,d):
        return (-tengeante_y/norme,tengeante_x/norme)
    else:
        return (tengeante_y/norme,-tengeante_x/norme)
    

def upsampling_dict(dict,L):
    new_dict = {}
    for key in dict.keys():
        new_dict[(key[0]*L,key[1]*L)] = (dict[key][0]*L,dict[key][1]*L)
    return new_dict

def filterlow(im,L): 
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    mask=(abs(XX)<tx/(2*L)) & (abs(YY)<ty/(2*L))
    imtf=np.fft.fft2(imt)
    imtf[~mask]=0
    return np.real(np.fft.ifft2(imtf))

def filtergauss(im,L):
    """applique un filtre passe-bas gaussien. coupe approximativement a f0/2L"""
    (ty, tx) = im.shape
    imt = np.float32(im.copy())
    pi = np.pi

    XX = np.fft.fftfreq(tx).reshape(1, tx) * tx
    YY = np.fft.fftfreq(ty).reshape(ty, 1) * ty

    sig = (tx * ty)**0.5 / (2 * L * (pi**0.5))

    mask = np.exp(-(XX**2 + YY**2) / (2 * sig**2))

    imtf = np.fft.fft2(imt) * mask
    return np.real(np.fft.ifft2(imtf))