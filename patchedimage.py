from utilities import *

class PatchedImage():
    def __init__(self, filename, size):
        self.img = plt.imread(filename).copy().astype(np.float64)

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.size = size

        self.patch_flat = None
        self.tree = None #leaf_size à changer ? en fonction de la taille de l'image
        self.current_mask = None

        self.zone = self.set_zone() # Tout le patch doit etre dans la zone ?  #0 = target region, 1 = frontière, 2 = source region
        self.working_patch = (-1,-1)
        self.masque = None

        self.confidence = np.ones(self.img.shape)
        self.data = np.zeros(self.img.shape)
        self.priority = np.zeros(self.img.shape)

        self.gradient = np.zeros((2,self.height,self.width))
    
    def periodic_boundary(self, start_row, end_row, start_col, end_col):
        row_indices = np.arange(start_row, end_row) % self.height
        col_indices = np.arange(start_col, end_col) % self.width
        return self.img[np.ix_(row_indices, col_indices)]

    

    def patch_boundaries(self,coord):
        k,l = coord
        return k-self.size,k+self.size+1,l-self.size,l+self.size+1
    
    def set_patch_flat(self):
        """
        img_padded = np.pad(self.img, ((self.size, self.size), (self.size, self.size)), mode='constant')
        shape = (self.length - 2 * self.size, self.width - 2 * self.size, 2 * self.size + 1, 2 * self.size + 1)
        strides = img_padded.strides * 2
        sub_matrices = np.lib.stride_tricks.as_strided(img_padded, shape=shape, strides=strides)
        tab = sub_matrices.reshape(-1, (2 * self.size + 1) ** 2)
        return tab
        """
        tab = []
        for i in range(self.size,self.height-self.size):
            for j in range(self.size,self.width-self.size):
                #tab.append(np.concatenate([np.array(self.img[i-self.size:i+self.size+1,j-self.size:j+self.size+1]).flatten(),np.ones(((self.size*2+1)**2,))]))
                tab.append(np.array(self.img[i-self.size:i+self.size+1,j-self.size:j+self.size+1]).flatten())
        return np.array(tab)
    
    def set_zone(self):
        return np.ones(self.img.shape)*2 #tout est source au debut
    
    def set_working_patch(self,coord):
        self.working_patch = coord

    def outlines_target(self,size):
        noyau = np.ones((size,size))/size**2
        masque_conv = convolve2d(self.masque, noyau, mode='same')
        return np.argwhere((masque_conv< 0.75) & (masque_conv>0.1))

    def set_masque(self,masque,leaf_size): #1 pour le masque, 0 pour le reste
        for i in range(self.height):
            for j in range(self.width):
                if masque[i,j] == 1:
                    self.img[i,j] = np.nan
                    
        self.masque = masque
        self.zone = self.zone*(1-masque)
        outlines = self.outlines_target(2)
        self.zone[outlines[:,0],outlines[:,1]] = 1
        self.patch_flat = self.set_patch_flat()
        self.tree = BallTree(self.patch_flat, leaf_size=leaf_size,metric=self.masked_distance) # de taille image avec 1 pour le masque, 0 pour le reste

    def get_patch(self,coord):
        i,j = coord
        return self.img[i-self.size:i+self.size+1,j-self.size:j+self.size+1]
    
    def set_patch(self,coord,patch):
        i,j = coord
        self.img[i-self.size:i+self.size+1,j-self.size:j+self.size+1] = patch
    
    def set_priorities(self): #tres tres long pour le moment (a optimiser)
        if self.working_patch == (-1, -1):
            for i in range(self.size,self.height-self.size): #+1 ?
                for j in range(self.size,self.width-self.size): #+1 ?
                    if self.zone[i,j] == 1:
                        conf = self.set_confidence_patch((i,j))
                        dat = self.set_data_patch((i,j))
                        self.priority[i,j] = conf*dat
        else:
            k,l = self.working_patch
            conf = self.set_confidence_patch((k,l))
            dat = self.set_data_patch((k,l))
            self.priority[k,l] = conf*dat

    def find_max_priority(self):
        mask = (self.zone == 1) #& condition # on cherche le max dans la frontière
        masked_priority = self.priority[mask]
        max_index = np.argmax(masked_priority)
        original_indices = np.argwhere(mask)[max_index]
        return tuple(original_indices)

    def set_confidence_patch(self,coord):
        k,l = coord
        somme = 0
        for i in range(k-self.size,k+self.size+1):
            for j in range(l-self.size,l+self.size+1):
                if self.zone[i,j] == 2:
                    somme += self.confidence[i,j]
        res = somme/(self.size*2+1)**2
        self.confidence[coord] = res
        return res
    
    def mean_valid_gradient(self, i, j,):
        #Compute the mean gradients of the valid (non-NaN) neighbors around a given point (i, j) in a 2D arrayfor both x and y directions.
        grad_y, grad_x = self.gradient
        
        valid_gradients_y = []
        valid_gradients_x = []

        # Check the neighbors' gradients; considering the Moore neighborhood (8 surrounding cells)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if di == 0 and dj == 0:
                    continue  # Skip the center point itself
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    # Append valid (non-NaN) gradients from each dimension
                    if not np.isnan(grad_y[ni, nj]):
                        valid_gradients_y.append(grad_y[ni, nj])
                    if not np.isnan(grad_x[ni, nj]):
                        valid_gradients_x.append(grad_x[ni, nj])

        # Calculate the mean of valid gradients for each axis
        mean_gradient_y = np.mean(valid_gradients_y)
        mean_gradient_x = np.mean(valid_gradients_x)

        return (mean_gradient_y, mean_gradient_x)

    def set_gradient_patch(self, coord):
        a,b,c,d = self.patch_boundaries(coord)
        im_patch = self.periodic_boundary(a-1,b+1,c-1,d+1)
        xgrad, ygrad = np.gradient(im_patch)
        self.gradient[0][a:b,c:d] = xgrad[1:b-a+1,1:d-c+1]
        self.gradient[1][a:b,c:d] = ygrad[1:b-a+1,1:d-c+1]

        for i in range(a,b):
            for j in range(c,d):
                if self.zone[i,j]==1:
                    xgrad, ygrad = self.mean_valid_gradient(i, j)
                    self.gradient[0][i,j] = xgrad
                    self.gradient[1][i,j] = ygrad

    def compute_normal(self, coord):
        i,j = coord
        if self.zone[i,j] != 1:
            raise ValueError('trying to calculate normal vector not in frontier')
        
        border_neighbors = []
        target_neighbors = (-1,-1)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if di == 0 and dj == 0:
                    continue  # Skip the center point itself
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    if self.zone[ni,nj] != 1:
                        border_neighbors.append((ni, nj))
                    elif self.zone[ni,nj] == 0:
                        target_neighbors = (ni,nj)

        border_neighbors=sorted(border_neighbors)
        a,b,c,d = border_neighbors[0][0],border_neighbors[0][1],border_neighbors[-1][0],border_neighbors[-1][1]
        x,y = target_neighbors

        tengeante_x,tengeante_y = a-c,b-d

        if y - ((d-b)/(c-a)*(x-a)+b) > 0:
            return (-tengeante_y,tengeante_x)
        else:
            return (tengeante_y,-tengeante_x)

    def set_data_patch(self,coord):
        k,l = coord
        self.set_gradient_patch(coord)
        a,b,c,d = self.patch_boundaries(coord)
        for i in range(a,b):
            for j in range(c,d):
                if self.zone[i,j] != 0:
                    grad_ij = (self.gradient[0][i,j],self.gradient[1][i,j])
                    normal_ij = self.compute_normal(coord)
                    self.data[i,j] = np.dot(orthogonal_vector(grad_ij),normal_ij)/255
        return self.data[k,l]
    
    def masked_distance(self, patch1, patch2):
        if self.current_mask is not None:
            mask = self.current_mask
        else:
            mask = np.ones_like(patch1)
        dist = np.linalg.norm((patch1 - patch2) * mask)
        return dist
    
    def find_nearest_patch(self,coord): #renvoie le complementaire du masque
        patch = self.get_patch((coord[0],coord[1]))
        p_size = 2*self.size+1
        patchs = self.patch_flat
        masque = (patch != 0).flatten()
        self.current_mask = masque
        ind = self.tree.query([patch.flatten()], k=2,return_distance=False)
        return (patchs[ind[0,1]][:(p_size**2)]* (1-masque)).reshape((p_size,p_size))
    
    def reconstruction(self,coord): #un patch
        pato = self.find_nearest_patch(coord)
        recons = self.get_patch(coord)+pato
        self.set_patch(coord,recons)

    def show_patch(self,coord = None):
        if coord == None:
            coord = self.working_patch
        k,l = coord
        pat = self.get_patch(coord)
        plt.imshow(pat, cmap='gray',vmin=0,vmax=255)
        plt.title(f"Priority : {self.priority[k,l]:.3f}")
        plt.show()

    def show_img(self):
        #fig, ax = plt.subplots()
        plt.imshow(self.img, cmap='gray',vmin=0,vmax=255)
        #if self.masque != None:
        #    x1,y1 = self.masque[0]
        #    x2,y2 = self.masque[1]
            #square = patches.Rectangle((y1,x1),y2-y1,x2-x1,linewidth=1,edgecolor='r',facecolor='r')
            #ax.add_patch(square)
            #ax.plot([self.masque[0][1],self.masque[1][1],self.masque[1][1],self.masque[0][1],self.masque[0][1]],[self.masque[0][0],self.masque[0][0],self.masque[1][0],self.masque[1][0],self.masque[0][0]],color=(0,1,0))
        #plt.show()

    def show_patch_in_img(self, coord = None):
        if coord == None:
            coord = self.working_patch
        k,l = coord
        #contours = self.outlines_patch(coord)
        plt.imshow(self.img, cmap='gray',vmin=0,vmax=255)
        plt.plot([l-self.size,l+self.size,l+self.size,l-self.size,l-self.size],[k-self.size,k-self.size,k+self.size,k+self.size,k-self.size],color=(0,1,0))