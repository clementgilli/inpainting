from .utilities import *
from .draw import *
from copy import deepcopy 

def masked_dist(patch1, patch2, mask):
    #patch1 = deepcopy(patch1)
    #patch2 = deepcopy(patch2)

    patch1[np.isnan(patch1)] = -1
    patch2[np.isnan(patch2)] = -1
    for i in range(len(patch1)):
        for j in range(len(patch1[0])):
            if patch1[i,j] == -1:
                patch2[i,j] = -1
    dist = np.linalg.norm((patch1 - patch2) * (1-mask))
    return dist #* hellinger_distance(patch1,patch2)

class PatchedImage():
    def __init__(self, filename, size, search_mode="Full"): 
        self.filename = filename
        self.search_mode = search_mode #Full or Local
        
        self.img = plt.imread(filename).copy().astype(np.float64)
        if len(self.img.shape) == 3:
            self.img = rgb2gray(self.img)

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.size = size

        self.zone = np.ones(self.img.shape)*2 # Tout le patch doit etre dans la zone ?  #0 = target region, 1 = frontière, 2 = source region
        self.working_patch = (-1,-1)
        self.masque = None

        self.confidence = np.ones(self.img.shape)
        self.data = np.zeros(self.img.shape)
        self.priority = np.zeros(self.img.shape)

        self.gradient = np.full((2, self.height, self.width), np.nan)
    
    def periodic_boundary(self, start_row, end_row, start_col, end_col):
        row_indices = np.arange(start_row, end_row) % self.height
        col_indices = np.arange(start_col, end_col) % self.width
        return self.img[np.ix_(row_indices, col_indices)]

    def patch_boundaries(self,coord):
        k,l = coord
        return k-self.size,k+self.size+1,l-self.size,l+self.size+1
    
    def search_zone(self,size_search):
        return search_zone_compiled(self.height,self.width,self.size,self.masque,size_search)
    
    def set_patch_flat(self):
        tab = []
        coords = []
        if self.search_mode == "Local":
            a,b = self.search_zone(int(5*np.sqrt(min(self.width,self.height)))) #totally arbitrary
        else:
            a,b = (self.size,self.height-self.size),(self.size,self.width-self.size)
        self.search_zone_coord = (a,b)
        for i in range(a[0],a[1]):#range(self.size,self.height-self.size):
            for j in range(b[0],b[1]):#range(self.size,self.width-self.size):
                patch = np.array(self.img[i-self.size:i+self.size+1, j-self.size:j+self.size+1])
                #patch[np.isnan(patch)] = 0
                tab.append(patch.flatten())
                coords.append((i, j))
        self.patch_coords = np.array(coords)
        return np.array(tab)
        
    def outlines_target(self):
        return np.array(outlines_target_compiled(self.height,self.width,self.masque))

    def set_masque(self,draw=True,masque=None): #1 pour le masque, 0 pour le reste
        #self.img = self.img*(1-masque)
        if draw:
            self.img, self.masque = draw_on_image(self.filename)
        else:
            self.masque = masque
            self.img[self.masque == 1] = -1
        self.zone = self.zone*(1-self.masque)
        outlines = self.outlines_target()
        self.zone[outlines[:,0],outlines[:,1]] = 1
        self.patch_flat = self.set_patch_flat()

    def get_patch(self,coord):
        a,b,c,d = self.patch_boundaries(coord)
        return self.img[a:b,c:d]
    
    def set_patch(self,coord,patch): 
        #a,b,c,d = self.patch_boundaries(coord)
        #for i in range(a, b):
        #    for j in range(c, d):
        #        if np.isnan(self.img[i, j]):
        #            self.img[i, j] = patch[i - a, j - c]
        i,j = coord
        self.img[i-self.size:i+self.size+1,j-self.size:j+self.size+1] = patch
    
    def set_priorities(self): #tres tres long pour le moment (a optimiser)
        if self.working_patch == (-1, -1):
            a,b = self.search_zone_coord
            for i in range(a[0],a[1]): #range(self.size,self.height-self.size):
                for j in range(b[0],b[1]): #range(self.size,self.width-self.size):
                    if self.zone[i,j] == 1:
                        conf = self.set_confidence_patch((i,j))
                        dat = self.set_data_patch((i,j))
                        self.priority[i,j] = conf*dat
                        #sigma = 0.5
                        #self.priority[i,j] = conf*np.exp(dat/(2*sigma**2))
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
                if self.zone[i,j] != 0:
                    somme += self.confidence[i,j]
        res = somme/(self.size*2+1)**2
        self.confidence[coord] = res
        return res
    
    def mean_valid_gradient(self, i, j,):
        #Compute the mean gradients of the valid (non-NaN) neighbors around a given point (i, j) in a 2D arrayfor both x and y directions.
        grad_y, grad_x = self.gradient
        try:
            return mean_valid_gradient_compiled(i,j,grad_y,grad_x,self.height,self.width)
        except ZeroDivisionError:
            return 0,0
        
    def set_gradient_patch(self, coord):
        if self.zone[coord] == 0:
            raise ValueError("Trying to calculate the gradient in the target region")
        k,l = coord
        a,b,c,d = k-1,k+2,l-1,l+2 #self.patch_boundaries(coord)
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
        return compute_normal_compiled(coord, self.zone, self.height, self.width)

    def set_data_patch(self,coord):
        k,l = coord
        self.set_gradient_patch(coord)
        a,b,c,d = self.patch_boundaries(coord)
        if self.zone[k,l] == 1:
            grad_ij = (self.gradient[0][k,l],self.gradient[1][k,l])
            normal_ij = self.compute_normal((k,l))
            self.data[k,l] = np.dot(orthogonal_vector(grad_ij),normal_ij)/255
        #for i in range(a,b):
        #    for j in range(c,d):
        #        if self.zone[i,j] != 0:
        #            grad_ij = (self.gradient[0][i,j],self.gradient[1][i,j])
        #            normal_ij = self.compute_normal(coord)
        #            self.data[i,j] = np.dot(orthogonal_vector(grad_ij),normal_ij)/255
        return self.data[k,l]
    
    def masked_distance(self, patch1, patch2):
        if self.current_mask is not None:
            mask = self.current_mask
        else:
            mask = np.isnan(patch1).astype(int)
        return masked_dist(patch1,patch2,mask)
        #dist = np.linalg.norm((patch1 - patch2) * mask)
        #return dist
    
    def reconstruction(self,coord): #un patch
        k,l = coord
        coord_min = self.find_nearest_patch(coord)
        pato = self.get_patch(coord_min)
        recons = self.get_patch(coord).copy()
        for i in range(len(pato)):
            for j in range(len(pato[0])):
                if recons[i,j] == -1 and pato[i,j] != -1:
                    recons[i,j] = pato[i,j]
                    self.zone[k-self.size+i,l-self.size+j] = 2
                    self.masque[k-self.size+i,l-self.size+j] = 0
        #plt.imshow(recons,cmap="gray",vmin=0,vmax=255)
        #plt.show()
        self.set_patch(coord,recons)
        #probablement à changer ce q'il y a en dessous
        outlines = self.outlines_target()
        if outlines.size != 0:
            self.zone[self.zone == 1] = 2
            self.zone[outlines[:,0],outlines[:,1]] = 1

    def show_patch(self,coord = None):
        if coord == None:
            coord = self.working_patch
        k,l = coord
        pat = self.get_patch(coord)
        plt.imshow(pat, cmap='gray',vmin=0,vmax=255)
        plt.title(f"Priority : {self.priority[k,l]:.3f}")
        plt.show()

    def show_img(self,search_zone=False):
        plt.imshow(self.img, cmap='gray',vmin=0,vmax=255)
        plt.colorbar()
        if search_zone:
            c1,c2 = self.search_zone_coord
            plt.plot([c2[0],c2[0],c2[1],c2[1],c2[0]],[c1[0],c1[1],c1[1],c1[0],c1[0]],color=(0,1,0))
        plt.show()

    def show_patch_in_img(self, coord = None):
        if coord == None:
            coord = self.working_patch
        k,l = coord
        #contours = self.outlines_patch(coord)
        plt.imshow(self.img, cmap='gray',vmin=0,vmax=255)
        plt.plot([l-self.size,l+self.size,l+self.size,l-self.size,l-self.size],[k-self.size,k-self.size,k+self.size,k+self.size,k-self.size],color=(0,1,0))

    def reconstruction_auto(self, iter_max = np.inf, display_img = False, display_iter = False, save_result=False ,save_gif=False):
        i = 0
        t1 = time()
        while len(self.zone[self.zone==0]) != 0 and i < iter_max:
            self.set_priorities()
            coord = self.find_max_priority()
            self.reconstruction(coord)
            i += 1
            if display_iter:
                print(f"iteration {i} done")
            
            if display_img: # and i%10 == 0:
                #self.show_img()
                plt.imshow(self.zone, cmap='gray')
                plt.show()
                #cv2.imshow('frame',self.img/255)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            if save_gif:
                cv2.imwrite(f"gifs/{i}.jpg", self.img)
            #self.show_img()
        cv2.destroyAllWindows()
        t2 = time()
        print(f"Reconstruct in {t2-t1:.3f} sec")
        if save_result:
            cv2.imwrite(f"results/res.jpg", self.img)
        if save_gif:
            images = []
            filenames = sorted((int(fn.split(".")[0]) for fn in os.listdir('./gifs/') if fn.endswith('.jpg')))
            for filename in filenames:
                images.append(imageio.imread("./gifs/"+str(filename)+".jpg"))
                os.remove("./gifs/"+str(filename)+".jpg")
            imageio.mimsave('./gifs/test.gif', images)
        #cv2.imshow('Result',self.img/255)
        cv2.waitKey(0)

    def reconstruct_with_dict(self,coords):
        if coords == {}:
            raise ValueError("No coordinates saved")
        for coord in coords:
            k,l = coords[coord]
            pat = self.get_patch(coords[coord]) * (1-self.masque[k-self.size:k+self.size+1,l-self.size:l+self.size+1])
            self.set_patch(coord,pat)

            #self.masque[k-self.size:k+self.size+1,l-self.size:l+self.size+1] = 0
            #self.zone[k-self.size:k+self.size+1,l-self.size:l+self.size+1] = 2

    def find_nearest_patch(self,coord): #renvoie le complementaire du masque
        a,b = self.search_zone_coord
        patch_list1 = np.array([[(i,j) for i in range(a[0], a[1])] for j in range(b[0],b[1])]).flatten()
        patch_list = [(patch_list1[2*i],patch_list1[2*i+1]) for i in range(len(patch_list1)//2)]

        m = np.inf
        patch1 = self.get_patch(coord)
        patch_min = self.get_patch(coord)

        masque = np.isnan(patch1).astype(int)

        for k_coord in patch_list:
            if self.zone[k_coord[0],k_coord[1]] == 2:
                patch2 = self.get_patch(k_coord)
                d = masked_dist(patch1, patch2, masque)
                if d < m:
                    m = d
                    coord_min = k_coord
        return coord_min

            