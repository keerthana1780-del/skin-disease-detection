import os
from dotenv import load_dotenv

load_dotenv()

import tensorflow as tf
import numpy as np

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

app = Flask(__name__)

# ==============================
# Telegram
# ==============================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

telegram_app = Application.builder().token(TOKEN).build()


# ==============================
# Load CNN Model
# ==============================

MODEL_PATH = "models/skin_disease_cnn_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)


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
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Fungal Infection"
]


# ==============================
# Website Upload Folder
# ==============================

UPLOAD_FOLDER = "sample_data"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# Website
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    description = ""

    if request.method == "POST":

        description = request.form.get("description", "")

        image = request.files.get("image")

        if image and image.filename:

            filename = secure_filename(image.filename)

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(image_path)

            img = tf.keras.utils.load_img(
                image_path,
                target_size=IMG_SIZE
            )

            img_array = tf.keras.utils.img_to_array(img)

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

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
# Telegram /start
# ==============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "👋 Welcome to Skin Disease Detection Bot!\n\n"
        "📸 Send a skin image to get an AI prediction.\n\n"
        "📌 Use /help for commands.\n"
        "ℹ️ Use /about for project information."
    )


# ==============================
# Telegram /help
# ==============================

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


# ==============================
# Telegram /about
# ==============================

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


# ==============================
# Telegram Photo
# ==============================

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

        img = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        img_array = tf.keras.utils.img_to_array(
            img
        )

        img_array = tf.expand_dims(
            img_array,
            0
        )

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        confidence = (
            np.max(predictions[0]) * 100
        )

        disease = class_names[
            predicted_index
        ]

        await update.message.reply_text(
            f"🔍 Prediction Result\n\n"
            f"🦠 Disease: {disease}\n"
            f"📊 Confidence: {confidence:.2f}%\n\n"
            f"⚠️ This is an AI model prediction, "
            f"not a medical diagnosis."
        )

    except Exception as e:

        print("Telegram Error:", e)

        await update.message.reply_text(
            "❌ Sorry, I couldn't process this image."
        )


# ==============================
# Telegram Handlers
# ==============================

telegram_app.add_handler(
    CommandHandler("start", start)
)

telegram_app.add_handler(
    CommandHandler("help", help_command)
)

telegram_app.add_handler(
    CommandHandler("about", about_command)
)

telegram_app.add_handler(
    MessageHandler(filters.PHOTO, handle_photo)
)


# ==============================
# Telegram Webhook
# ==============================

@app.route("/telegram-webhook", methods=["POST"])
async def telegram_webhook():

    data = request.get_json(force=True)

    update = Update.de_json(
        data,
        telegram_app.bot
    )

    await telegram_app.process_update(update)

    return "OK"


# ==============================
# Health Check
# ==============================

@app.route("/health")
def health():

    return "Skin Disease AI is running!"


# ==============================
# Run
# ==============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )