import numpy as np
import matplotlib.pyplot as plt

def gradient(ar):
    grad_i, grad_j = np.gradient(ar)
    return np.array([[(grad_i[i, j], grad_j[i, j]) for j in range(grad_i.shape[1])] for i in range(grad_j.shape[0])])

class PatchedImage():
    def __init__(self, filename, size):
        self.img = plt.imread(filename)
        self.length = self.img.shape[0]
        self.width = self.img.shape[1]
        self.size = size
        self.zone = self.set_zone() # Tout le patch doit etre dans la zone ?  #0 = target region, 1 = frontière, 2 = source region
        self.working_patch = (-1,-1)

        self.confidence = np.ones(self.img.shape)
        self.data = np.zeros(self.img.shape)
        self.priority = np.zeros(self.img.shape)

        grad_i, grad_j = np.gradient(self.img)
        self.gradient = np.array([[(grad_i[i, j], grad_j[i, j]) for j in range(grad_i.shape[1])] for i in range(grad_j.shape[0])])
    
    def set_zone(self):
        return np.ones(self.img.shape)*2 #tout est source au debut
    
    def set_working_patch(self,coord):
        self.working_patch = coord

    def outlines_patch(self,coord):
        k,l = coord
        img = self.img[k-self.size:k+self.size+1,l-self.size:l+self.size+1]
        outlines = np.array([img[0,:],img[-1,:],img[:,0],img[:,-1]])
        return np.concatenate(outlines)

    
    def set_priorities(self): #tres tres long pour le moment (a optimiser)
        if self.working_patch == (-1, -1):
            for i in range(self.size,self.length-self.size+1):
                for j in range(self.size,self.width-self.size+1):
                    if self.zone[i,j] != 0:
                        #self.set_priority_patch((i,j))
                        conf = self.set_confidence_patch((i,j))
                        dat = self.set_data_patch((i,j))
                        self.priority[i,j] = conf*dat
        else:
            k,l = self.working_patch
            conf = self.set_confidence_patch((k,l))
            dat = self.set_data_patch((k,l))
            self.priority[k,l] = conf*dat

    def set_confidence_patch(self,coord):
        k,l = coord
        somme = 0
        for i in range(k-self.size,k+self.size):
            for j in range(l-self.size,l+self.size):
                if self.zone[i,j] == 2:
                    somme += self.confidence[i,j]
        res = somme/(self.size*2+1)**2
        self.confidence[coord] = res
        return res

    def set_gradient_patch(self, coord):
        k,l = coord
        self.gradient[k-self.size:k+self.size+1,l-self.size:l+self.size+1] = gradient(self.img[k-self.size:k+self.size+1,l-self.size:l+self.size+1])

    def set_data_patch(self,coord):
        i,j = coord
        
        #gradient = np.gradient(self.img[i-1:i+2,j-1:j+2])
        #grad_patch = np.array([gradient[0][1,1],gradient[1][1,1]])
        #à faire
        
        return 1

    def show_patch(self,coord = None):
        if coord == None:
            coord = self.working_patch

        k,l = coord
        img = self.img[k-self.size:k+self.size+1,l-self.size:l+self.size+1]
        plt.imshow(img, cmap='gray')
        plt.title(f"Priority : {self.priority[k,l]:.3f}")
        plt.show()

    def show_img(self):
        plt.imshow(self.img, cmap='gray')
        plt.show()

    def show_patch_in_img(self, coord = None):
        if coord == None:
            coord = self.working_patch
        k,l = coord
        contours = self.outlines_patch(coord)
        plt.imshow(self.img, cmap='gray')
        plt.plot([l-self.size,l+self.size,l+self.size,l-self.size,l-self.size],[k-self.size,k-self.size,k+self.size,k+self.size,k-self.size],'r')