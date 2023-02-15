from typing import Tuple

from tensorflow import keras


def build_cnn_model(input_shape: Tuple[int, int, int] = (800, 600, 3), num_classes: int = 2) -> keras.Model:
    """
    Build a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape (tuple of 3 ints, optional): The shape of the input images, default is (800, 600, 3).
    - num_classes (int, optional): The number of classes the model will predict, default is 1.

    Returns:
    - model (keras.Model): A compiled CNN model.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.01)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
