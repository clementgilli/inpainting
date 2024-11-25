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
            cv2.circle(mask, (x, y), brush_size, 1, -1)
            cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, 1, -1)
        cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)

def update_brush_size(val):
    global brush_size
    brush_size = val

def draw_on_image(image_path):
    global brush_size

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Erreur: Impossible de charger l'image.")
        return
    
    if len(image.shape) == 2:  # Image en niveaux de gris
        is_gray = True
        image_float = image.astype(np.float32)
        overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:  # Image RGB
        is_gray = False
        image_float = image.astype(np.float32)
        overlay_image = image.copy()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_with_mouse, param=(image_float, overlay_image, mask))

    cv2.createTrackbar('Taille du pinceau', 'image', brush_size, 50, update_brush_size)
    print("Press 'q' when you are done.")
    while True:
        alpha = 0.6 
        if is_gray:
            blended_image = cv2.addWeighted(overlay_image, alpha, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)
        else:
            blended_image = cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0)

        cv2.imshow('image', blended_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    modified_image = image.copy()
    
    if is_gray:
        modified_image[mask == 1] = 0
    else:
        modified_image[mask == 1] = [0, 0, 0]
        #modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    return modified_image, mask

# utilisation :
#image_path = 'images/lena.tif'
#image, mask = draw_on_image(image_path)