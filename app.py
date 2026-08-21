import os
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

# ==========================================
# BASE DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# TELEGRAM BOT
# ==========================================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

telegram_app = None

if TOKEN:
    telegram_app = Application.builder().token(TOKEN).build()
else:
    print("WARNING: TELEGRAM_BOT_TOKEN is not set.")
    print("Telegram bot will be disabled.")

# ==========================================
# CNN MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "skin_disease_cnn_v2.keras"
)

print("Loading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")

IMG_SIZE = (224, 224)

# ==========================================
# DISEASE CLASSES
# ==========================================

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

# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "sample_data"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================
# WEBSITE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    description = ""

    if request.method == "POST":

        description = request.form.get(
            "description",
            ""
        )

        image = request.files.get("image")

        if image and image.filename:

            filename = secure_filename(
                image.filename
            )

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(image_path)

            try:

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

                index = int(
                    np.argmax(predictions[0])
                )

                prediction = class_names[index]

                confidence = round(
                    float(predictions[0][index]) * 100,
                    2
                )

            except Exception as e:

                print("Prediction Error:", e)

                prediction = "Unable to process image"
                confidence = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description
    )


# ==========================================
# TELEGRAM /START
# ==========================================

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


# ==========================================
# TELEGRAM /HELP
# ==========================================

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


# ==========================================
# TELEGRAM /ABOUT
# ==========================================

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


# ==========================================
# TELEGRAM PHOTO
# ==========================================

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

        confidence = (
            float(np.max(predictions[0])) * 100
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


# ==========================================
# TELEGRAM HANDLERS
# ==========================================

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


# ==========================================
# TELEGRAM WEBHOOK
# ==========================================

@app.route(
    "/telegram-webhook",
    methods=["POST"]
)
async def telegram_webhook():

    if telegram_app is None:
        return "Telegram bot is not configured", 503

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
            e
        )

        return "Webhook error", 500


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return "Skin Disease AI is running!"


# ==========================================
# RUN
# ==========================================

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