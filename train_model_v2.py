import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# =============================
# SETTINGS
# =============================

DATASET_PATH = "../IMG_CLASSES"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

# =============================
# GET ALL IMAGE FILES
# =============================

class_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
])

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

# =============================
# CREATE CLASS-WISE SPLIT
# =============================

train_paths = []
train_labels = []

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

    print(f"{class_name}: {len(images)} images")

    train_images, val_images = train_test_split(
        images,
        test_size=0.2,
        random_state=SEED
    )

    train_paths.extend(train_images)
    train_labels.extend([label] * len(train_images))

    val_paths.extend(val_images)
    val_labels.extend([label] * len(val_images))

print("\nTotal training images:", len(train_paths))
print("Total validation images:", len(val_paths))

# =============================
# CREATE TF DATASETS
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


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

train_ds = train_ds.shuffle(
    len(train_paths),
    seed=SEED
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# =============================
# CREATE CNN MODEL
# =============================

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Data augmentation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Normalize
    layers.Rescaling(1./255),

    # CNN Block 1
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    # CNN Block 2
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    # CNN Block 3
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    # Fully connected
    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    # 10 classes
    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# =============================
# COMPILE
# =============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model created successfully!")

# =============================
# TRAIN
# =============================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =============================
# SAVE MODEL
# =============================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model.save(
    "models/skin_disease_cnn_v2.keras"
)

# =============================
# SAVE HISTORY
# =============================

with open(
    "results/training_history_v2.json",
    "w"
) as f:

    json.dump(
        history.history,
        f
    )

print("\n=============================")
print("Training completed!")
print("=============================")

print("Model saved:")
print("models/skin_disease_cnn_v2.keras")

print("\nTraining history saved:")
print("results/training_history_v2.json")