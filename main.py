from patchedimage import PatchedImage
from patchedimage_color import PatchedImageColor
from resampling import Resampling
from utilities import *
import sys

if len(sys.argv) < 3:
    print("Usage: python main.py <path_image> <patch_size> <optional:downsampling_factor>")
    sys.exit(1)

assert sys.argv[2].isdigit(), "The patch size must be an integer\nUsage: python main.py <path_image> <patch_size> <optional:downsampling_factor>"

if len(sys.argv) == 4:
    assert sys.argv[3].isdigit(), "The downsampling factor must be an integer\nUsage: python main.py <path_image> <patch_size> <optional:downsampling_factor>"
    re = Resampling(sys.argv[1],int(sys.argv[2]))
    re.one_level(int(sys.argv[3]))

else:

    try :
        imgp = PatchedImageColor(sys.argv[1],size=int(sys.argv[2]),search_mode="Local")
    except ValueError:
        imgp = PatchedImage(sys.argv[1],size=int(sys.argv[2]),search_mode="Local")

    print("====Initialisation====")
    imgp.set_masque(leaf_size=max(imgp.width,imgp.height))
    imgp.set_priorities()
    #imgp.show_img(search_zone=True)

    print("====Reconstruction====")
    imgp.reconstruction_auto(display_img=True,save_result=True,save_gif=False,show_result=True)