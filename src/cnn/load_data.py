import os
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from preprocessing import preprocess_image


def crop_center(img: np.ndarray) -> np.ndarray:
    """
    Cut a square size 224x224 from the center of an image.

    Parameters:
    - img (numpy array): The input image.

    Returns:
    - crop (numpy array): The cropped image.
    """
    height, width = img.shape[0], img.shape[1]
    start_x = (width - 224) // 2
    start_y = (height - 224) // 2
    crop = img[start_y: start_y + 224, start_x: start_x + 224]
    return crop


def read_data(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the data from the folder containing a CSV file and images.

    Parameters:
    - folder_path (str): The path of the folder.

    Returns:
    - (x, y) (tuple of 2 numpy arrays): x is the feature array, y is the label array.
    """
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(os.path.join(folder_path, 'labels.csv'))

    # Initialize arrays to store the images and labels
    x, y = [], []

    # Loop over the rows of the dataframe
    for index, row in df.iterrows():
        # Read the image file
        image = cv2.imread(os.path.join(folder_path, row['filename']))

        # Crop the image
        image = crop_center(image)

        # Preprocess the image
        image = preprocess_image(image)

        # Append the image and label to the arrays
        x.append(image)
        y.append(row['label'])

    # Convert the arrays to numpy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y
