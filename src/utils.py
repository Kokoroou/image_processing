import cv2
import numpy as np
from matplotlib import pyplot as plt


def build_histogram(image: np.ndarray = None):
    gray_img = None

    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray_img = image.copy()

    cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.hist(gray_img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale picture')
    plt.show()
