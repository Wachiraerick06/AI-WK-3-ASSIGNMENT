import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def build_and_train_model() -> tf.keras.Model:
    # Load and preprocess MNIST
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Define a simple CNN similar to the notebook
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train briefly to keep it fast
    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=2)
    return model


def main() -> None:
    model = build_and_train_model()
    model_path = os.path.join(os.getcwd(), "mnist_model.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()


