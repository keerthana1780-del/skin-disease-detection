import os
import tensorflow as tf
import numpy as np

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

MODEL_PATH = "models/skin_disease_cnn.keras"
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Skin Disease Detection Bot!\n\n"
        "📸 Send a skin image to get an AI prediction.\n\n"
        "📌 Use /help to see available commands.\n"
        "ℹ️ Use /about to learn about this project."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Available Commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show available commands\n"
        "/about - About the project\n\n"
        "📸 Send a skin image to get a prediction."
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🩺 Skin Disease Detection Bot\n\n"
        "🤖 This project uses a CNN-based deep learning model "
        "to classify skin images into different disease categories.\n\n"
        "🧠 Model: Convolutional Neural Network (CNN)\n"
        "🖼️ Image Size: 224 × 224 pixels\n"
        "📊 Classes: 10\n\n"
        "📸 Send a skin image to test the model.\n\n"
        "⚠️ This bot provides an AI model prediction only "
        "and is not a medical diagnosis."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "🔍 Analyzing the image... Please wait."
        )

        photo = update.message.photo[-1]

        file = await context.bot.get_file(photo.file_id)

        image_path = "sample_data/telegram_image.jpg"

        await file.download_to_drive(image_path)

        img = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        img_array = tf.keras.utils.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(predictions[0])

        confidence = np.max(predictions[0]) * 100

        disease = class_names[predicted_index]

        await update.message.reply_text(
            f"🔍 Prediction Result\n\n"
            f"🦠 Disease: {disease}\n"
            f"📊 Confidence: {confidence:.2f}%\n\n"
            f"⚠️ This is an AI model prediction, "
            f"not a medical diagnosis."
        )

    except Exception as e:

        print("Error:", e)

        await update.message.reply_text(
            "❌ Sorry, I couldn't process this image.\n\n"
            "Please try sending another image."
        )


def main():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CommandHandler("help", help_command)
    )

    app.add_handler(
        CommandHandler("about", about_command)
    )

    app.add_handler(
        MessageHandler(filters.PHOTO, handle_photo)
    )

    print("🤖 Bot is running...")

    app.run_polling()


if __name__ == "__main__":
    main()