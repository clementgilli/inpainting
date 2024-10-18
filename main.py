from patchedimage import PatchedImage
from patchedimage_color import PatchedImageColor
from utilities import *
import sys

if len(sys.argv) != 3:
    print("Usage: python main.py <path_image> <patch_size>")
    sys.exit(1)
    
try :
    imgp = PatchedImageColor(sys.argv[1],size=int(sys.argv[2]))
except ValueError:
    imgp = PatchedImage(sys.argv[1],size=int(sys.argv[2]))

print("====Initialisation====")
imgp.set_masque(leaf_size=max(imgp.width//2,imgp.height//2))
imgp.set_priorities()

print("====Reconstruction====")

imgp.reconstruction_auto(display_img=True,save=False)

imgp.show_img()