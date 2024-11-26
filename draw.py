import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
brush_size = 20

def draw_with_mouse(event, x, y, flags, param):
    global ix, iy, drawing, brush_size

    image_with_drawing, overlay_image, mask = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(image_with_drawing, (x, y), brush_size, 0, -1)
            cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), brush_size, 1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(image_with_drawing, (x, y), brush_size, 0, -1)
        cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), brush_size, 1, -1)


def update_brush_size(val):
    global brush_size
    brush_size = val


def draw_on_image(image_path, save_mask_path=None):
    global brush_size

    image = cv2.imread(image_path)
    if image is None:
        print("Erreur: Impossible de charger l'image.")
        return

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_float = image.astype(np.float32)
    overlay_image = image.copy() 
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_with_mouse, param=(image_float, overlay_image, mask))
    cv2.createTrackbar('Taille du pinceau', 'image', brush_size, 50, update_brush_size)

    while True:
        alpha = 0.6
        blended_image = cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0)
        cv2.imshow('image', blended_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quitter
            break
        elif key == ord('s'):
            if save_mask_path:
                # Sauvegarde du masque dans un fichier .npy
                np.save(save_mask_path, mask)
                print(f"Masque sauvegardé dans {save_mask_path}")
            else:
                print("Chemin de sauvegarde non spécifié.")
        elif key == ord('p'):  # Sauvegarder comme image PNG
            mask_png_path = save_mask_path.replace(".npy", ".png") if save_mask_path else "mask.png"
            cv2.imwrite(mask_png_path, mask * 255)
            print(f"Masque sauvegardé comme image dans {mask_png_path}")

    cv2.destroyAllWindows()

    return image_float, mask

# utilisation :
#image_path = 'images/lena.tif'
#image, mask = draw_on_image(image_path)