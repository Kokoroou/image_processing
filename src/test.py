import cv2

from utils import *
from constants import DATA_DIR


if __name__ == '__main__':
    image_filename = 'low_contrast_3.jpg'
    image_filepath = DATA_DIR / image_filename

    # image = cv2.imread(str(image_filepath), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_filepath))

    width = image.shape[1]
    height = image.shape[0]

    new_height = 750
    new_width = int(new_height / height * width)

    image = cv2.resize(image, dsize=(new_width, new_height))

    build_histogram(image)
    cv2.imshow('Image', image)
    cv2.waitKey()
    # print(image.dtype)

    new_image = enhance_contrast(image)
    # print(new_image.dtype)
    build_histogram(new_image)
    cv2.imshow('New Image', new_image)
    cv2.waitKey()

