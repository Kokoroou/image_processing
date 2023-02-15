import cv2
import numpy as np


def convert_to_dft(input_image: np.ndarray) -> np.ndarray:
    """
    Computes the magnitude spectrum of an input image using the Discrete Fourier Transform.

    Args:
        input_image: A 2D or 3D NumPy array representing the input image in BGR format.

    Returns:
        A 2D NumPy array representing the magnitude spectrum of the input image.

    Raises:
        ValueError: If the input image is not a 2D or 3D NumPy array.
    """

    if len(input_image.shape) not in (2, 3):
        raise ValueError("Input image must be a 2D or 3D NumPy array")

    # Convert input image to grayscale
    if len(input_image.shape) == 3:
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = input_image

    # Compute the 2D DFT of the input image
    dft = np.fft.fft2(gray_image)

    # Shift the zero-frequency component to the center of the spectrum
    dft_shifted = np.fft.fftshift(dft)

    # Compute the magnitude spectrum of the DFT
    magnitude_spectrum = np.log(np.abs(dft_shifted))

    # Normalize the magnitude spectrum to the range [0, 255]
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return magnitude_spectrum


if __name__ == "__main__":
    image_path = "/home/misa/Workspace/Company/Research/Image_Processing/data/raw/id_card/12_frontside_19_real.jpg"

    image = cv2.imread(image_path)

    dft_output = convert_to_dft(image)

    cv2.imshow("Original", image)
    cv2.imshow("DFT", dft_output)
    cv2.waitKey()
