import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

model = tf.keras.models.load_model(
    "saved_models/best_model_augmentation.keras"
)

print("Model loaded successfully!")

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "BacterialLeafBlight",
    "BrownSpot",
    "LeafSmut"
]

def load_split(file_path):

    image_paths = []
    labels = []

    with open(file_path, "r") as f:
        for line in f:
            path, label = line.strip().split(",")
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels

test_paths, test_labels = load_split(
    "splits/test.txt"
)
print("Test samples: ", len(test_paths))

def load_image(image_path, label):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(
        image,
        channels = 3
    )

    image = tf.image.resize(
        image,
        (IMG_HEIGHT, IMG_WIDTH)
    )

    image = image / 255.0

    return image, label

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_dataset = test_dataset.map(load_image)

test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)

#evaluating
test_loss, test_accuracy = model.evaluate(
    test_dataset, verbose = 1
)

print("\n========== Test Results ==========")
print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss : {test_loss:.4f}")

#PREDICTING
predictions = model.predict(test_dataset)

predicted_labels = np.argmax(
    predictions,
    axis = 1
)

true_labels = np.array(test_labels)

#compute metrics
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average = "weighted"
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average = "weighted"
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average = "weighted"
)

print("\n========== Evaluation Metrics ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names = CLASS_NAMES
    )
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels = CLASS_NAMES
)

fig, ax = plt.subplots(figsize = (6,6))

disp.plot(ax = ax, cmap = "Blues")
plt.title("Confusion Matrix")
plt.savefig(
    "results/confusion_matrix.png",
    dpi = 300
)

plt.show()