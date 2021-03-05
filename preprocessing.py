import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def preprocess_func(img):
    img = cv2.normalize(
        img, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    return img
