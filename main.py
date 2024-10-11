from patchedimage import PatchedImage
from utilities import *
import sys

imgp = PatchedImage(sys.argv[1],size=int(sys.argv[2]))
print("====Initialisation====")
imgp.set_masque(leaf_size=max(imgp.width//2,imgp.height//2),draw=True)
imgp.set_priorities()

print("====Reconstruction====")
for k in range(1000):
    try:
        i,j = imgp.find_max_priority()
    except ValueError:
        print("priority error")
        break
    imgp.reconstruction((i,j))
    try:
        imgp.set_priorities()
    except ValueError:
        print("pti bug")

imgp.show_img()
plt.show()