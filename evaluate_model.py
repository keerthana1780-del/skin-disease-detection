import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Settings
# -----------------------------
DATASET_PATH = "../IMG_CLASSES"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load validation dataset
# -----------------------------
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_data.class_names

print("Classes:")
print(class_names)

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model(
    "models/skin_disease_cnn.keras"
)

print("Model loaded successfully!")

# -----------------------------
# Get predictions
# -----------------------------
y_true = []
y_pred = []

for images, labels in val_data:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_true == y_pred)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Display and save
# -----------------------------
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(12, 10))

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Skin Disease CNN - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved successfully!")
print("Saved at: results/confusion_matrix.png")

from collections import Counter

print("\nValidation images per class:")

counts = Counter(y_true)

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name} -> {counts[i]} images")