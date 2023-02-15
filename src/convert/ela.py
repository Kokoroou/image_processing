import os

import cv2
import numpy as np


def convert_to_ela(image: np.ndarray, quality: int = 70) -> np.ndarray:
    """
    Convert a given color image to its Error Level Analysis (ELA).

    Parameters:
    image (np.ndarray): A NumPy array representing a color image.
    quality (int): The quality level to use when saving the image for ELA. Defaults to 90.

    Returns:
    np.ndarray: A NumPy array representing the ELA image.
    """

    # Save image array to a temporary file at a given quality level
    temp_file = "temp.jpg"
    cv2.imwrite(temp_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Load the saved image and calculate the ELA
    temp_image = cv2.imread(temp_file)

    # Calculate the absolute difference between the original image and the saved/reloaded image
    ela_image = cv2.absdiff(image, temp_image)

    # Calculate the maximum absolute difference in the ELA image
    max_diff = np.max(ela_image)

    # If the maximum absolute difference is 0, set it to 1 to avoid division by zero errors
    if max_diff == 0:
        max_diff = 1

    # Calculate a scaling factor to map the maximum absolute difference to the maximum intensity value of 255
    scale = 255.0 / max_diff

    # Scale the ELA image using the scaling factor and convert to unsigned 8-bit integers
    ela_image = np.multiply(scale, ela_image).astype('uint8')

    # Remove the temporary file
    os.remove(temp_file)

    return ela_image


if __name__ == "__main__":
    image_path = "/home/misa/Workspace/Company/Research/Image_Processing/data/raw/id_card/12_frontside_19_real.jpg"

    image = cv2.imread(image_path)

    ela_output = convert_to_ela(image, 70)

    cv2.imshow("Original", image)
    cv2.imshow("ELA", ela_output)
    cv2.waitKey()
