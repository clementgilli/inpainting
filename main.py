from patchedimage import PatchedImage
from utilities import *
import sys

imgp = PatchedImage(sys.argv[1],size=int(sys.argv[2]))
print("====Initialisation====")
imgp.set_masque(leaf_size=max(imgp.width//2,imgp.height//2),draw=True)
imgp.set_priorities()

print("====Reconstruction====")

imgp.reconstruction_auto(display_img=True)

imgp.show_img()
plt.show()