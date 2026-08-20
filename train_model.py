import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json

DATASET_PATH = "../IMG_CLASSES"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("Classes:")
print(class_names)

# Improve data pipeline speed
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Create CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Data augmentation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Normalize pixels
    layers.Rescaling(1./255),

    # CNN layers
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Improved CNN model created successfully!")

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

print("Training completed successfully!")

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Save model
model.save("models/skin_disease_cnn.keras")

# Save training history
with open("results/training_history.json", "w") as f:
    json.dump(history.history, f)

print("Model saved successfully!")
print("Training history saved successfully!")