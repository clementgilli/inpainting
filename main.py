from src.patchedimage import PatchedImage
from src.patchedimage_color import PatchedImageColor
from src.resampling import Resampling
from src.utilities import *
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Process an image with optional parameters.")
    parser.add_argument("path_image", type=str, help="Path to the input image file.")
    parser.add_argument("patch_size", type=int, help="Size of the patches (integer).")
    parser.add_argument(
        "--mode",
        choices=["Local", "Full"],
        default="Local",
        help="Search mode to use. Defaults to 'Local'.",
    )
    parser.add_argument(
        "--dsf",
        type=int,
        help="Downsampling factor to use. Must be an integer.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="Path to a .npy file containing a mask.",
    )

    args = parser.parse_args()
    print(args)

    path_image = args.path_image
    patch_size = args.patch_size
    search_mode = args.mode
    downsampling_factor = args.dsf
    mask_file = args.mask

    try:
        if downsampling_factor is not None:
            print(f"Resampling mode: Applying resampling with factor {downsampling_factor}")
            re = Resampling(path_image, patch_size)
            re.one_level(downsampling_factor)
        else:
            print(f"Patched image mode: Using search mode '{search_mode}'")
            try:
                imgp = PatchedImageColor(path_image, size=patch_size, search_mode=search_mode)
            except ValueError:
                imgp = PatchedImage(path_image, size=patch_size, search_mode=search_mode)

            print("==== Initialisation ====")
            if mask_file:
                print(f"Using mask from file: {mask_file}")
                imgp.set_masque(leaf_size=max(imgp.width, imgp.height), draw=False, masque=np.load(mask_file))
            else:
                imgp.set_masque(leaf_size=max(imgp.width, imgp.height))

            imgp.set_priorities()
            # imgp.show_img(search_zone=True)

            print("==== Reconstruction ====")
            imgp.reconstruction_auto(display_img=True, save_result=True, save_gif=False, show_result=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()