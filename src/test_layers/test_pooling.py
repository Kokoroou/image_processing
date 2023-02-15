import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model


def apply_max_pooling(image_input, pool_size):
    height, width, channel = image_input.shape

    # Create a Keras input with the same shape as the image
    input_layer = Input(shape=(height, width, channel))

    # # Add a convolutional layer with a single filter
    # # This is just to add a channel dimension to the image
    # conv_layer = Conv2D(filters=1, kernel_size=1, padding='same')(input_layer)

    # Add a max pooling layer with the specified pool size
    pool_layer = MaxPooling2D(pool_size=pool_size)(input_layer)

    # Create a Keras model with the input and output layers
    model = Model(inputs=input_layer, outputs=pool_layer)

    # Convert image to float and normalize to range [0, 1]
    image = image_input.astype(np.float32) / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Apply max pooling using the Keras model
    pooled_image = model.predict(image)

    # Convert back to uint8 and remove the channel dimension
    pooled_image = (pooled_image[0] * 255).astype(np.uint8)

    return pooled_image


def apply_average_pooling(image_input, pool_size):
    height, width, channel = image_input.shape

    # Create a Keras input with the same shape as the image
    input_layer = Input(shape=(height, width, channel))

    # # Add a convolutional layer with a single filter
    # # This is just to add a channel dimension to the image
    # conv_layer = Conv2D(filters=1, kernel_size=1, padding='same')(input_layer)

    # Add an average pooling layer with the specified pool size
    pool_layer = AveragePooling2D(pool_size=pool_size)(input_layer)

    # Create a Keras model with the input and output layers
    model = Model(inputs=input_layer, outputs=pool_layer)

    # Convert image to float and normalize to range [0, 1]
    image = image_input.astype(np.float32) / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Apply average pooling using the Keras model
    pooled_image = model.predict(image)

    # Convert back to uint8 and remove the channel dimension
    pooled_image = (pooled_image[0] * 255).astype(np.uint8)

    return pooled_image


if __name__ == "__main__":
    # Read image with cv2
    image_original = cv2.imread('/home/misa/Workspace/Company/Research/Image_Processing/data/raw/unlabel/image1.jpg',
                                cv2.IMREAD_COLOR)

    # Apply pooling with a pool size of 2 using Keras
    image_max_pooled = apply_max_pooling(image_original, (2, 2))
    image_average_pooled = apply_average_pooling(image_original, (2, 2))

    # Display the original and pooled images side by side
    cv2.imshow('Original', image_original)
    cv2.imshow('Max Pooled', image_max_pooled)
    cv2.imshow('Average Pooled', image_average_pooled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
