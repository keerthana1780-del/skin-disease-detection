import os
import urllib.request
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# TELEGRAM BOT
# =========================================================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

telegram_app = None

if TOKEN:
    telegram_app = Application.builder().token(TOKEN).build()
    print("Telegram bot token found.")
else:
    print("WARNING: TELEGRAM_BOT_TOKEN is not set.")
    print("Telegram bot will be disabled.")

# =========================================================
# HUGGING FACE MODEL
# =========================================================

HF_MODEL_URL = (
    "https://huggingface.co/"
    "keerthana1780/skin-disease-cnn/"
    "resolve/main/"
    "models/skin_disease_cnn_v2.keras"
)

# IMPORTANT:
# Do NOT use /app/models/... here.
# Render will download the model to /tmp.

MODEL_PATH = "/tmp/skin_disease_cnn_v2.keras"


# =========================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# =========================================================

def download_model():

    print("==========================================")
    print("CHECKING CNN MODEL")
    print("==========================================")

    # If model already exists, use it
    if os.path.exists(MODEL_PATH):

        size = os.path.getsize(MODEL_PATH)

        print("Existing model found.")
        print("Model size:", size, "bytes")

        if size > 100_000_000:
            print("Model already downloaded.")
            return

        print("Existing model is too small.")
        print("Removing corrupted model...")

        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    print("==========================================")
    print("DOWNLOADING CNN MODEL")
    print("==========================================")

    print("Hugging Face URL:")
    print(HF_MODEL_URL)

    try:

        request = urllib.request.Request(
            HF_MODEL_URL,
            headers={
                "User-Agent": "Skin-Disease-AI"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=900
        ) as response:

            with open(
                MODEL_PATH,
                "wb"
            ) as output:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output.write(chunk)

        size = os.path.getsize(
            MODEL_PATH
        )

        print("Downloaded model size:")
        print(size, "bytes")

        if size < 100_000_000:

            raise RuntimeError(
                "Downloaded model is smaller than expected."
            )

        print("==========================================")
        print("CNN MODEL DOWNLOADED SUCCESSFULLY")
        print("==========================================")

    except Exception as e:

        print("==========================================")
        print("MODEL DOWNLOAD ERROR")
        print("==========================================")

        print(repr(e))

        if os.path.exists(MODEL_PATH):

            try:
                os.remove(MODEL_PATH)
            except Exception:
                pass

        raise


# =========================================================
# DOWNLOAD MODEL
# =========================================================

download_model()


# =========================================================
# LOAD MODEL
# =========================================================

print("==========================================")
print("LOADING CNN MODEL")
print("==========================================")

print("Model path:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "CNN model was not downloaded."
    )

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("==========================================")
    print("CNN MODEL LOADED SUCCESSFULLY")
    print("==========================================")

except Exception as e:

    print("==========================================")
    print("MODEL LOAD ERROR")
    print("==========================================")

    print(repr(e))

    raise


# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_SIZE = (224, 224)


# =========================================================
# DISEASE CLASSES
# =========================================================

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


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "sample_data"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    disease = class_names[
        predicted_index
    ]

    confidence = (
        float(
            predictions[0][
                predicted_index
            ]
        ) * 100
    )

    return disease, confidence


# =========================================================
# WEBSITE HOME
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    confidence = None

    description = ""

    if request.method == "POST":

        description = request.form.get(
            "description",
            ""
        )

        image = request.files.get(
            "image"
        )

        if image and image.filename:

            filename = secure_filename(
                image.filename
            )

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(
                image_path
            )

            try:

                prediction, confidence = (
                    predict_image(
                        image_path
                    )
                )

            except Exception as e:

                print(
                    "Prediction Error:",
                    repr(e)
                )

                prediction = (
                    "Unable to process image"
                )

                confidence = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description
    )


# =========================================================
# TELEGRAM /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "👋 Welcome to Skin Disease Detection Bot!\n\n"

        "📸 Send a skin image to get an AI prediction.\n\n"

        "📌 Use /help for commands.\n"

        "ℹ️ Use /about for project information."

    )


# =========================================================
# TELEGRAM /HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "📌 Available Commands:\n\n"

        "/start - Start the bot\n"

        "/help - Show available commands\n"

        "/about - About the project\n\n"

        "📸 Send a skin image to get a prediction."

    )


# =========================================================
# TELEGRAM /ABOUT
# =========================================================

async def about_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🩺 Skin Disease Detection Bot\n\n"

        "🤖 CNN-based deep learning project "
        "for skin image classification.\n\n"

        "🧠 Model: CNN\n"

        "🖼️ Image Size: 224 × 224\n"

        "📊 Classes: 10\n\n"

        "⚠️ This is an AI model prediction, "
        "not a medical diagnosis."

    )


# =========================================================
# TELEGRAM PHOTO
# =========================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    try:

        await update.message.reply_text(
            "🔍 Analyzing the image... Please wait."
        )

        photo = update.message.photo[-1]

        file = await context.bot.get_file(
            photo.file_id
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            "telegram_image.jpg"
        )

        await file.download_to_drive(
            image_path
        )

        disease, confidence = (
            predict_image(
                image_path
            )
        )

        await update.message.reply_text(

            f"🔍 Prediction Result\n\n"

            f"🦠 Disease: {disease}\n"

            f"📊 Confidence: {confidence:.2f}%\n\n"

            f"⚠️ This is an AI model prediction, "
            f"not a medical diagnosis."

        )

    except Exception as e:

        print(
            "Telegram Error:",
            repr(e)
        )

        await update.message.reply_text(
            "❌ Sorry, I couldn't process this image."
        )


# =========================================================
# TELEGRAM HANDLERS
# =========================================================

if telegram_app:

    telegram_app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "about",
            about_command
        )
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )


# =========================================================
# TELEGRAM WEBHOOK
# =========================================================

@app.route(
    "/telegram-webhook",
    methods=["POST"]
)
async def telegram_webhook():

    if telegram_app is None:

        return (
            "Telegram bot is not configured",
            503
        )

    try:

        data = request.get_json(
            force=True
        )

        update = Update.de_json(
            data,
            telegram_app.bot
        )

        await telegram_app.process_update(
            update
        )

        return "OK"

    except Exception as e:

        print(
            "Webhook Error:",
            repr(e)
        )

        return (
            "Webhook error",
            500
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return "Skin Disease AI is running!"


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )