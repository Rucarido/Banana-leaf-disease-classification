import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset_path = "rice_leaf_diseases"

image_paths = []
labels = []

classes = sorted(os.listdir(dataset_path))

for label, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_path, class_name)
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        image_paths.append(image_path)
        labels.append(label)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

print("Training :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing :", len(test_paths))

# Create a folder to store split files
os.makedirs("splits", exist_ok=True)


def save_split(filename, paths, labels):
    with open(filename, "w") as f:
        for path, label in zip(paths, labels):
            f.write(f"{path},{label}\n")


save_split("splits/train.txt", train_paths, train_labels)
save_split("splits/validation.txt", val_paths, val_labels)
save_split("splits/test.txt", test_paths, test_labels)

print("\nDataset splits saved successfully!")