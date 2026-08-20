import tensorflow as tf
import numpy as np

MODEL_PATH = "models/skin_disease_cnn.keras"
IMAGE_PATH = "sample_data/test.jpg"

IMG_SIZE = (224, 224)

class_names = [
    "Eczema",
    "Warts Molluscum and other Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Fungal Infection"
]

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print("\nPrediction:")
print("Disease:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")