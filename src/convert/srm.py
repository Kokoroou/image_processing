# TODO: Read paper and implement real function: https://www.isihome.ir/freearticle/ISIHome.ir-26078.pdf
import cv2
import numpy as np
from scipy import signal


def srm_encode(img: np.ndarray, message: str, alpha: float = 0.1, w: int = 3) -> np.ndarray:
    """
    Embeds a secret message into an image using Steganalysis Rich Models (SRM).

    Parameters:
    img: The input image as a numpy array.
    message: The secret message to be embedded into the image.
    alpha: The strength of the embedding. Default is 0.1.
    w: The window size for the feature extraction. Default is 3.

    Returns:
    The encoded image as a numpy array.
    """

    # Convert message to binary
    bits = np.unpackbits(np.array([ord(c) for c in message], dtype=np.uint8))

    # Extract features from image
    x = img.astype(float)
    m, n, c = x.shape
    features = np.zeros((m - w + 1, n - w + 1, w ** 2))
    for i in range(w):
        for j in range(w):
            patch = x[i:m - w + i + 1, j:n - w + j + 1]
            features[:, :, i * w + j] = signal.convolve2d(patch, np.ones((w, w)), mode='valid')

    # Embed message into features
    alpha *= np.std(features)
    for i, bit in enumerate(bits):
        features[:, :, i] += alpha * bit

    # Synthesize image from features
    y = np.zeros((m, n))
    for i in range(w):
        for j in range(w):
            y[i:m - w + i + 1, j:n - w + j + 1] += features[:, :, i * w + j]

    # Normalize and clip image values
    y -= np.min(y)
    y /= np.max(y)
    y *= 255
    y = np.clip(np.round(y), 0, 255)

    return y.astype(np.uint8)


if __name__ == "__main__":
    image_path = "/home/misa/Workspace/Company/Research/Image_Processing/data/raw/id_card/12_frontside_19_real.jpg"

    image = cv2.imread(image_path)

    srm_output = srm_encode(image, "What")

    cv2.imshow("Original", image)
    cv2.imshow("SRM", srm_output)
    cv2.waitKey()
