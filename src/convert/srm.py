import cv2
import numpy as np
from scipy.signal import convolve2d


def convert_to_srm(image: np.ndarray) -> np.ndarray:
    """
    Applies Steganalysis Rich Models (SRM) to an image to extract features that can be used to filter for hidden data.

    Args:
        image: A NumPy array representing an image to be processed.

    Returns:
        A NumPy array representing the processed image.
    """
    # Split the image into separate color channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Define the three 5x5 kernels
    kernel_blue = np.multiply(1.0 / 2, np.array([[0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 1, -2, 1, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]], dtype=np.float32))
    kernel_green = np.multiply(1.0 / 12, np.array([[-1, 2, -2, 2, -1],
                                                   [2, -6, 8, -6, 2],
                                                   [-2, 8, -12, 8, -2],
                                                   [2, -6, 8, -6, 2],
                                                   [-1, 2, -2, 2, -1]], dtype=np.float32))
    kernel_red = np.multiply(1.0 / 4, np.array([[0, 0, 0, 0, 0],
                                                [0, -1, 2, -1, 0],
                                                [0, 2, -4, 2, 0],
                                                [0, -1, 2, -1, 0],
                                                [0, 0, 0, 0, 0]], dtype=np.float32))

    # Apply the kernels to each color channel separately
    filtered_blue = convolve2d(blue_channel, kernel_blue, mode='same')
    filtered_green = convolve2d(green_channel, kernel_green, mode='same')
    filtered_red = convolve2d(red_channel, kernel_red, mode='same')

    # Clip the values to a valid range and convert to an 8-bit unsigned integer
    filtered_blue = np.clip(np.multiply(255.0, filtered_blue, dtype=np.float32), 0, 255).astype(np.uint8)
    filtered_green = np.clip(np.multiply(255.0, filtered_green, dtype=np.float32), 0, 255).astype(np.uint8)
    filtered_red = np.clip(np.multiply(255.0, filtered_red, dtype=np.float32), 0, 255).astype(np.uint8)

    # Merge the filtered color channels back into a single image
    filtered_image = cv2.merge((filtered_red, filtered_green, filtered_blue))

    # Convert the image to grayscale
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)

    return filtered_image


if __name__ == "__main__":
    # Load an example image
    image_path = "/home/misa/Workspace/Company/Research/Image_Processing/data/raw/id_card/12_frontside_15_screen.jpg"
    input_image = cv2.imread(image_path)

    # Apply the custom filter to the image using the combined kernel
    output_filtered_image = convert_to_srm(input_image)

    # cv2.imshow("New", cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("New", output_filtered_image)

    cv2.waitKey()
