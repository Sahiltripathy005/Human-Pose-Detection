import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple CNN model for pose estimation
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(34)  # 17 keypoints (x, y) pairs
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and save the model
model = create_model()
model.save("pose_model.h5")
