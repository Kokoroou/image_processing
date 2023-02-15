from pathlib import Path

from cnn.build_model import build_cnn_model
from cnn.load_data import read_data

if __name__ == "__main__":
    main_path = Path(__file__).parent.parent.resolve()
    data_path = main_path / 'data' / 'raw' / 'cnn'

    # Load image and label
    x_train, y_train = read_data(str(Path(data_path, "train")))
    x_val, y_val = read_data(str(Path(data_path, "val")))
    x_test, y_test = read_data(str(Path(data_path, "test")))

    # Build model
    model = build_cnn_model()

    # Training
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # Evaluation
    score = model.evaluate(x_test, y_test, batch_size=32)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
