import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d
from sklearn.neighbors import BallTree
import numba

@numba.jit(nopython=True,fastmath=True)
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

@numba.jit(nopython=True)
def orthogonal_vector(v):
    return np.array([-v[1], v[0]])

@numba.jit(nopython=True,fastmath=True)
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
        valid_gradients_y = [0] #à changer
    if valid_gradients_x == []:
        valid_gradients_x = [0] #à changer
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
        print("bug à",coord)
        raise ValueError('no target neighbors found')

    border_neighbors=sorted(border_neighbors)
    a,b,c,d = border_neighbors[0][0],border_neighbors[0][1],border_neighbors[-1][0],border_neighbors[-1][1]
    x,y = target_neighbors

    tengeante_x,tengeante_y = a-c,b-d
    norme = (tengeante_x**2+tengeante_y**2)**0.5
        
    if below_line(x,y, a,b, c,d):
        return (-tengeante_y/norme,tengeante_x/norme)
    else:
        return (tengeante_y/norme,-tengeante_x/norme)
    
def neighbours(i,j):
    return [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]