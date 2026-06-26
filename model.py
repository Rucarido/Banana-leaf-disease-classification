import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():

    model = models.Sequential([

        # Input Layer
        layers.Input(shape=(224, 224, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(), #we use this instead of Flatten because flatten increases no. of training parameters which can
        #cause overfitting risks, while globalaveragepooling keeps the average and simplifies the model
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(3, activation='softmax')

    ])

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()