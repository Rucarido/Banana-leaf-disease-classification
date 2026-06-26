import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "rice_leaf_diseases"
classes = os.listdir(dataset_path)

print("Disease Classes:")

for disease in classes:
    print(disease)

print("\nNumber of Images:")

for disease in classes:

    disease_folder = os.path.join(dataset_path, disease)
    images = os.listdir(disease_folder)
    print(f"{disease}: {len(images)} images")

#Display one image from each class
plt.figure(figsize=(12,4))

for i, disease in enumerate(classes):

    folder = os.path.join(dataset_path, disease)
    image_name = os.listdir(folder)[0]
    image_path = os.path.join(folder, image_name)
    image = Image.open(image_path)
    plt.subplot(1,3,i+1)
    plt.imshow(image)
    plt.title(disease)
    plt.axis("off") 
plt.show()