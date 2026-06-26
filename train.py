import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import build_model
from augmentation import get_augmentation

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "BacterialLeafBlight",
    "BrownSpot",
    "LeafSmut"
]

data_augmentation = get_augmentation()

def load_split(file_path):

    image_paths = []
    labels = []

    with open(file_path, "r") as f:
        for line in f:
            path, label = line.strip().split(",")
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels

train_paths, train_labels = load_split("splits/train.txt")
val_paths, val_labels = load_split("splits/validation.txt")
test_paths, test_labels = load_split("splits/test.txt")

#small verification of loaded data
print("Training data: ", len(train_paths))
print("Validation data: ", len(val_paths))
print("Testing data: ", len(test_paths))

#load the images with their labels like (img1, 0) meaning img1 with label 0 is loaded
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

"""
#<_TensorSliceDataset ...> is expected
print(train_dataset)
print(validation_dataset)
print(test_dataset)
"""

"""
Image
↓
Read
↓
Resize
↓
Normalize
↓
Augmentation (added further after baseline)
↓
Shuffle
↓
Batch
↓
CNN

Tensorflow input pipeline is like this
"""

def load_image(image_path, label):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0

    return image, label

def augment_image(image, label):

    image = data_augmentation(image, training = True)
    return image, label

#mapping labels and data
train_dataset = train_dataset.map(load_image)
train_dataset = train_dataset.map(augment_image)
train_dataset = train_dataset.shuffle(
    buffer_size = len(train_paths),
    reshuffle_each_iteration = True
)
validation_dataset = validation_dataset.map(load_image)
validation_dataset = validation_dataset.cache()
test_dataset = test_dataset.map(load_image)
test_dataset = test_dataset.cache()

#batching the data
train_dataset = train_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#small optimization using prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

#verification of output
#for images, labels in train_dataset.take(1):
#    print("Image batch shape :", images.shape)
#    print("Label batch shape :", labels.shape)

plt.figure(figsize=(10,10))

#visualization
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(CLASS_NAMES[labels[i]])
        plt.axis("off")

plt.show()

#BUILDING MODEL
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(
    optimizer = optimizer,
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights = True
)

#storing best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved_models/best_model_augmentation.keras",
    monitor = "val_accuracy",
    save_best_only = True
)

history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 50, #50 during augmentation to help it learn better while in baseline it was 30
    callbacks = [early_stopping, checkpoint],
    shuffle = False
)

history_df = pd.DataFrame(history.history)
history_df.to_csv("results/training_history_augmented.csv", index=False)

print("\nTraining Completed Successfully!")