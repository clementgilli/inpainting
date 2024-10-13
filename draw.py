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
            cv2.circle(image_with_drawing, (x, y), brush_size, 0, -1) #ou -1 à la place 0
            #cv2.rectangle(image_with_drawing, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 0, -1) #ou -1 à la place 0
            cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)
            #cv2.rectangle(overlay_image, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 0, -1) #ou -1 à la place 0
            cv2.circle(mask, (x, y), brush_size, 1, -1)
            #cv2.rectangle(mask, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 1, -1) #ou -1 à la place 0

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(image_with_drawing, (x, y), brush_size, 0, -1)
        cv2.circle(overlay_image, (x, y), brush_size, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), brush_size, 1, -1)
        #cv2.rectangle(image_with_drawing, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 0, -1) #ou -1 à la place 0
        #cv2.rectangle(overlay_image, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 0, -1) #ou -1 à la place 0
        #cv2.rectangle(mask, (x-brush_size, y-brush_size), (x+brush_size,y+brush_size), 1, -1) #ou -1 à la place 0

def update_brush_size(val):
    global brush_size
    brush_size = val

def draw_on_image(image_path):
    global brush_size

    image = cv2.imread(image_path, 0)
    if image is None:
        print("Erreur: Impossible de charger l'image.")
        return

    image_float = image.astype(np.float32)  
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask = np.zeros_like(image, dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_with_mouse, param=(image_float, overlay_image, mask))
    cv2.createTrackbar('Taille du pinceau', 'image', brush_size, 50, update_brush_size)
    #cv2.setTrackbarMin('Taille du pinceau', 'image', 10)

    while True:
        alpha = 0.6
        blended_image = cv2.addWeighted(overlay_image, alpha, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)

        cv2.imshow('image', blended_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return image_float, mask

# utilisation :
#image_path = 'images/lena.tif'
#image, mask = draw_on_image(image_path)