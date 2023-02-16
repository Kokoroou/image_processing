from pathlib import Path

import cv2
import imutils

from dft import convert_to_dft
from ela import convert_to_ela
from srm import convert_to_srm


if __name__ == "__main__":
    image_dir = "/home/misa/Workspace/Company/Research/Image_Processing/data/raw/id_card/"

    for image_path in Path(image_dir).glob("*.jpg"):
        # Load an example image
        input_image = cv2.imread(str(image_path))

        input_image = imutils.resize(input_image, height=400)

        dft_output = convert_to_dft(input_image)
        srm_output = convert_to_srm(input_image)
        ela_output = convert_to_ela(input_image)

        cv2.imshow("Original", input_image)
        cv2.imshow("DFT", dft_output)
        cv2.imshow("SRM", srm_output)
        cv2.imshow("ELA", ela_output)

        cv2.waitKey()
