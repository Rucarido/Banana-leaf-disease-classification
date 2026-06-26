import os
import pandas as pd
import matplotlib.pyplot as plt

history  =  pd.read_csv("results/training_history_augmented.csv")

#Plotting accuracy
plt.figure(figsize = (8,5))

plt.plot(
    history["accuracy"],
    label = "Training Accuracy",
    linewidth = 2
)

plt.plot(
    history["val_accuracy"],
    label = "Validation Accuracy",
    linewidth = 2
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(
    "results/accuracy_curve.png",
    dpi = 300
)
plt.show()

#Plotting loss
plt.figure(figsize = (8,5))

plt.plot(
    history["loss"],
    label = "Training Loss",
    linewidth = 2
)

plt.plot(
    history["val_loss"],
    label = "Validation Loss",
    linewidth = 2
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(
    "results/loss_curve.png",
    dpi = 300
)
plt.show()