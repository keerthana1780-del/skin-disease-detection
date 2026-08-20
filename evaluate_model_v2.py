import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =============================
# SETTINGS
# =============================

DATASET_PATH = "../IMG_CLASSES"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# =============================
# GET CLASSES
# =============================

class_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
])

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

# =============================
# CREATE SAME VALIDATION SPLIT
# =============================

val_paths = []
val_labels = []

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for label, class_name in enumerate(class_names):

    class_folder = os.path.join(DATASET_PATH, class_name)

    images = [
        os.path.join(class_folder, file)
        for file in os.listdir(class_folder)
        if file.lower().endswith(image_extensions)
    ]

    _, validation_images = train_test_split(
        images,
        test_size=0.2,
        random_state=SEED
    )

    val_paths.extend(validation_images)
    val_labels.extend([label] * len(validation_images))

print("\nTotal validation images:", len(val_paths))

# =============================
# LOAD IMAGES
# =============================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label


val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# =============================
# LOAD V2 MODEL
# =============================

model = tf.keras.models.load_model(
    "models/skin_disease_cnn_v2.keras"
)

print("V2 model loaded successfully!")

# =============================
# PREDICTIONS
# =============================

y_true = []
y_pred = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =============================
# ACCURACY
# =============================

accuracy = np.mean(y_true == y_pred)

print("\n=============================")
print("V2 VALIDATION ACCURACY")
print("=============================")

print(f"{accuracy * 100:.2f}%")

# =============================
# CONFUSION MATRIX
# =============================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\nConfusion Matrix:")
print(cm)

# =============================
# SAVE CONFUSION MATRIX
# =============================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Skin Disease CNN V2 - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix_v2.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved successfully!")
print("results/confusion_matrix_v2.png")