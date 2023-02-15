from typing import Tuple

import cv2
import numpy as np


def preprocess_image(input_image: np.ndarray, resize_shape: Tuple[int, int] = (800, 600)) -> np.ndarray:
    def resize_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, target_shape)

    def normalize_image(image: np.ndarray) -> np.ndarray:
        return (image / 255.0) - 0.5

    # Resize the image
    resized_image = resize_image(input_image, resize_shape)

    # Normalize the image
    normalized_image = normalize_image(resized_image)

    return normalized_image
