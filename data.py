import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore

def load_and_preprocess_data():
    # Load MNIST dataset (handwritten digits)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)