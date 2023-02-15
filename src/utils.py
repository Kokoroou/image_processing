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


def _enhance_contrast_0(image: np.ndarray = None):
    """Linear stretching of intensity range"""
    max_matrix = np.full(shape=image.shape, fill_value=np.max(image))
    min_matrix = np.full(shape=image.shape, fill_value=np.min(image))

    input_range_matrix = np.subtract(max_matrix, min_matrix)
    output_range_matrix = np.full(shape=image.shape, fill_value=255)

    new_image = np.multiply(np.divide(np.subtract(image, min_matrix), input_range_matrix), output_range_matrix)
    new_image = new_image.astype('uint8')

    return new_image


def _enhance_contrast_1(image: np.ndarray = None, gamma: float = 1.5):
    """Gamma correction"""
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    new_image = cv2.LUT(image, table)

    return new_image


def _enhance_contrast_2(image: np.ndarray = None):
    """Histogram equalization"""
    equalized_img = None

    if len(image.shape) == 3:
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    elif len(image.shape) == 2:
        equalized_img = cv2.equalizeHist(image)

    return equalized_img


# --------------------------------------------------------------
enhance_contrast = _enhance_contrast_2
