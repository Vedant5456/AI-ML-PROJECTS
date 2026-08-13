import cv2
import numpy as np

def preprocess_signature(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)

    cropped = thresh[y:y+h, x:x+w]

    # structural validation
    white_pixels = np.sum(cropped == 255)
    density = white_pixels/(w*h)

    if density < 0.02 or density > 0.35:
        return None

    if w/h < 1.2:
        return None

    edges = cv2.Canny(cropped,50,150)

    if np.sum(edges>0) > 9000:
        return None

    resized = cv2.resize(cropped,(224,224))

    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    normalized = rgb/255.0

    return np.expand_dims(normalized,axis=0)