from patchedimage import PatchedImage
from patchedimage_color import PatchedImageColor
from utilities import *
import cv2
from draw import draw_on_image
from skimage.color import rgb2hsv, hsv2rgb

class Resampling():

    def __init__(self,filename,size):
        try :
            self.imgp = PatchedImageColor(filename,size,search_mode="Local")
            self.color = True
        except ValueError:
            self.imgp = PatchedImage(filename,size,search_mode="Local")
            self.color = False
        self.imgp.img, self.masque = draw_on_image(filename)
        self.imgp.masque = self.masque

    def one_level(self,M=2):
        self.imgp.show_img()
        if self.color:
            hsv = rgb2hsv(self.imgp.img)
            hsv = rgb2hsv(self.imgp.img)
            filter = filtergauss(hsv[:,:,2])
            hsv[:,:,2] = filter
            im = hsv2rgb(hsv)*255
        else:
            im = filtergauss(self.imgp.img)
        cv2.imwrite(f"undersampling.jpg", im[::M,::M])
        img = PatchedImage("undersampling.jpg",self.imgp.size//M,search_mode="Local")
        os.remove("undersampling.jpg")
        masque = self.masque[::M,::M]
        img.set_masque(128,False,masque)
        img.show_img()
        img.set_priorities()
        img.reconstruction_auto(display_img=False,save_result=False,save_gif=False)
        img.show_img()

        self.imgp.reconstruct_with_dict(upsampling_dict(img.save_coords,M))

        self.imgp.show_img()
