# 🌾 Rice Leaf Disease Prediction using CNN

A deep learning project developed using **TensorFlow** and **Convolutional Neural Networks (CNN)** to automatically classify rice leaf diseases from leaf images.

---

## 📌 Project Overview

Rice is one of the world's major food crops, and diseases affecting rice leaves can significantly reduce crop yield and quality. This project develops a Convolutional Neural Network (CNN) that classifies rice leaf images into three disease categories.

The project was implemented as part of the given problem statement:

> **Rice Leaf Disease Prediction using Machine Learning**

The model was first trained as a **Baseline CNN** and later improved using **Data Augmentation** to increase its ability to generalize on unseen data.

---

## 🎯 Objectives

- Develop a CNN model for rice leaf disease classification.
- Classify rice leaf images into three disease categories.
- Improve model performance using data augmentation.
- Evaluate the model using standard classification metrics.

---

## 🗂 Dataset

Dataset consists of **120 RGB images** divided into three classes.

| Class | Images |
|--------|--------|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 40 |

Dataset Split:

- Training : 80% (96 images)
- Validation : 10% (12 images)
- Testing : 10% (12 images)

Images are resized to:

```
224 × 224 × 3
```

---

## 📁 Project Structure

```
RiceDiseaseProject/

│
├── rice_leaf_diseases/
│   ├── BacterialLeafBlight/
│   ├── BrownSpot/
│   └── LeafSmut/
│
├── prepare_dataset.py
├── augmentation.py
├── model.py
├── train.py
├── evaluate.py
├── plot_history.py
│
├── saved_models/
│   ├── best_model.keras
│   └── best_model_augmented.keras
│
├── results/
│   ├── training_history.csv
│   ├── training_history_augmented.csv
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── ...
│
└── README.md
```

---

## 🧠 CNN Architecture

The CNN consists of:

```
Input (224×224×3)

↓

Conv2D (32 Filters)

↓

MaxPooling

↓

Conv2D (64 Filters)

↓

MaxPooling

↓

Conv2D (128 Filters)

↓

MaxPooling

↓

Flatten

↓

Dense (128)

↓

Dropout (0.5)

↓

Softmax Output (3 Classes)
```

Activation Function:

```
ReLU
```

Output Activation:

```
Softmax
```

---

## 🚀 Data Augmentation

To improve model generalization, the following augmentations were applied only to the training dataset:

- Horizontal Flip
- Random Rotation
- Random Zoom
- Random Contrast

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📈 Results

### Baseline CNN

| Metric | Value |
|---------|------:|
| Accuracy | 75.00% |
| Precision | 0.80 |
| Recall | 0.75 |
| F1-Score | 0.74 |

---

### CNN + Data Augmentation

| Metric | Value |
|---------|------:|
| Accuracy | **83.33%** |
| Precision | **0.85** |
| Recall | **0.83** |
| F1-Score | **0.83** |

The augmented model achieved better generalization and improved performance on unseen test images.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <repository-link>
```

---

### 2. Install dependencies

```bash
pip install tensorflow matplotlib pandas numpy scikit-learn
```

---

### 3. Place the dataset

```
rice_leaf_diseases/

├── BacterialLeafBlight/
├── BrownSpot/
└── LeafSmut/
```

---

### 4. Prepare Dataset

```bash
python prepare_dataset.py
```

---

### 5. Train the Model

```bash
python train.py
```

---

### 6. Evaluate the Model

```bash
python evaluate.py
```

---

### 7. Plot Training Curves

```bash
python plot_history.py
```

---

## 📌 Future Improvements

- Train on a larger rice disease dataset.
- Explore deeper CNN architectures.
- Apply transfer learning using EfficientNet or ResNet.
- Deploy the model as a web or mobile application.
- Add real-time disease prediction from camera images.

---

## 👨‍💻 Author

Project developed for the **Rice Leaf Disease Prediction** problem statement using **TensorFlow** and **Convolutional Neural Networks (CNN)**.

```

---
