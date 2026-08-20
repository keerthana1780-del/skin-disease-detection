from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# ==============================
# Load CNN Model
# ==============================

MODEL_PATH = "models/skin_disease_cnn_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ==============================
# Upload Folder
# ==============================

UPLOAD_FOLDER = "sample_data"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# Disease Classes
# ==============================

class_names = [
    "Eczema",
    "Warts Molluscum and other Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections"
]


# ==============================
# Home
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    description = ""

    if request.method == "POST":

        # Get description FIRST
        description = request.form.get("description", "")

        # Get image
        image = request.files.get("image")

        if image and image.filename:

            filename = secure_filename(image.filename)

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(image_path)

            # ==============================
            # Prepare Image
            # ==============================

            img = tf.keras.utils.load_img(
                image_path,
                target_size=(224, 224)
            )

            img_array = tf.keras.utils.img_to_array(img)

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ==============================
            # Prediction
            # ==============================

            predictions = model.predict(
                img_array,
                verbose=0
            )

            index = np.argmax(predictions[0])

            prediction = class_names[index]

            confidence = round(
                float(predictions[0][index]) * 100,
                2
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description
    )


# ==============================
# Run
# ==============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )