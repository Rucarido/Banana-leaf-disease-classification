import tensorflow as tf


def get_augmentation():

    data_augmentation = tf.keras.Sequential([

        # Random horizontal flip
        tf.keras.layers.RandomFlip(
            mode = "horizontal"
        ),

        # Small random rotation (about ±15°)
        tf.keras.layers.RandomRotation(
            factor = 0.08
        ),

        # Slight random zoom
        tf.keras.layers.RandomZoom(
            height_factor = 0.10,
            width_factor = 0.10
        ),

        # Small contrast variation
        tf.keras.layers.RandomContrast(
            factor = 0.10
        )

    ])

    return data_augmentation

#we use this inside model.py because:
#Only the training dataset should be augmented.
#Validation and test images should remain untouched.
#This keeps the model architecture identical between Experiment 1 and Experiment 2, so the only experimental difference is the input data.