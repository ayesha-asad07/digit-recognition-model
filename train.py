from data import load_and_preprocess_data
from model import build_model
import tensorflow as tf

def main():
    # Load and preprocess the data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # Build the model
    model = build_model()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=15, validation_split=0.1)

    # Save the model weights
    model.save_weights('model_weights.weights.h5')  # Updated filename

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    main()
    