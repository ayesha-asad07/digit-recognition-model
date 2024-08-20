import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flattening the output for the fully connected layers
        tf.keras.layers.Flatten(),
        
        # First dense layer with 64 neurons
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.5),
        
        # Second dense layer with 64 neurons
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output layer with 10 neurons (one for each digit class) and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model
