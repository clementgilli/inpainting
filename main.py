from patchedimage import PatchedImage
from patchedimage_color import PatchedImageColor
from utilities import *
import sys

if len(sys.argv) != 3:
    print("Usage: python main.py <path_image> <patch_size>")
    sys.exit(1)
    
try :
    imgp = PatchedImageColor(sys.argv[1],size=int(sys.argv[2]),search_mode="Local")
except ValueError:
    imgp = PatchedImage(sys.argv[1],size=int(sys.argv[2]),search_mode="Local")

print("====Initialisation====")
imgp.set_masque(leaf_size=max(imgp.width,imgp.height))
imgp.set_priorities()
#imgp.show_img(search_zone=True)

print("====Reconstruction====")
imgp.reconstruction_auto(display_img=True,save_result=True,save_gif=False)