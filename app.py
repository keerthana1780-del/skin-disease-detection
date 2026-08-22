import os
import urllib.request
import urllib.parse
import json
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from werkzeug.utils import secure_filename

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

RAILWAY_DOMAIN = os.environ.get(
    "RAILWAY_PUBLIC_DOMAIN",
    "skin-disease-detection-production.up.railway.app"
)

WEBHOOK_URL = (
    f"https://{RAILWAY_DOMAIN}/telegram-webhook"
)

if TOKEN:
    print("Telegram bot token found.")
    print("Telegram webhook URL:")
    print(WEBHOOK_URL)
else:
    print("WARNING: TELEGRAM_BOT_TOKEN is not set.")
    print("Telegram bot will be disabled.")


# =========================================================
# TELEGRAM API HELPER
# =========================================================

def telegram_api(method, data=None):

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    url = (
        f"https://api.telegram.org/bot{TOKEN}/{method}"
    )

    if data is None:
        data = {}

    encoded_data = urllib.parse.urlencode(
        data
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=encoded_data,
        method="POST"
    )

    with urllib.request.urlopen(
        req,
        timeout=60
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API error: {result}"
        )

    return result


# =========================================================
# SET TELEGRAM WEBHOOK
# =========================================================

def setup_telegram_webhook():

    if not TOKEN:
        return

    try:

        result = telegram_api(
            "setWebhook",
            {
                "url": WEBHOOK_URL,
                "allowed_updates": json.dumps(
                    [
                        "message",
                        "callback_query"
                    ]
                )
            }
        )

        print(
            "Telegram webhook configured successfully."
        )

        print(
            result
        )

    except Exception as e:

        print(
            "Telegram webhook setup error:",
            repr(e)
        )


# =========================================================
# HUGGING FACE MODEL
# =========================================================

HF_MODEL_URL = (
    "https://huggingface.co/"
    "keerthana1780/skin-disease-cnn/"
    "resolve/main/"
    "models/skin_disease_cnn_v2.keras"
)

MODEL_PATH = "/tmp/skin_disease_cnn_v2.keras"


# =========================================================
# DOWNLOAD MODEL
# =========================================================

def download_model():

    print("==========================================")
    print("CHECKING CNN MODEL")
    print("==========================================")

    if os.path.exists(MODEL_PATH):

        size = os.path.getsize(
            MODEL_PATH
        )

        print(
            "Existing model found."
        )

        print(
            "Model size:",
            size,
            "bytes"
        )

        if size > 100_000_000:

            print(
                "Model already downloaded."
            )

            return

        print(
            "Existing model is too small."
        )

        print(
            "Removing corrupted model..."
        )

        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    print("==========================================")
    print("DOWNLOADING CNN MODEL")
    print("==========================================")

    print(
        "Hugging Face URL:"
    )

    print(
        HF_MODEL_URL
    )

    try:

        req = urllib.request.Request(
            HF_MODEL_URL,
            headers={
                "User-Agent": "Skin-Disease-AI"
            }
        )

        with urllib.request.urlopen(
            req,
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

                    output.write(
                        chunk
                    )

        size = os.path.getsize(
            MODEL_PATH
        )

        print(
            "Downloaded model size:",
            size,
            "bytes"
        )

        if size < 100_000_000:

            raise RuntimeError(
                "Downloaded model is smaller than expected."
            )

        print("==========================================")
        print(
            "CNN MODEL DOWNLOADED SUCCESSFULLY"
        )
        print("==========================================")

    except Exception as e:

        print("==========================================")
        print("MODEL DOWNLOAD ERROR")
        print("==========================================")

        print(
            repr(e)
        )

        if os.path.exists(
            MODEL_PATH
        ):

            try:
                os.remove(
                    MODEL_PATH
                )
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

print(
    "Model path:",
    MODEL_PATH
)

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        "CNN model was not downloaded."
    )

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("==========================================")
    print(
        "CNN MODEL LOADED SUCCESSFULLY"
    )
    print("==========================================")

except Exception as e:

    print("==========================================")
    print("MODEL LOAD ERROR")
    print("==========================================")

    print(
        repr(e)
    )

    raise


# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_SIZE = (
    224,
    224
)


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

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_image(
    image_path
):

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
        np.argmax(
            predictions[0]
        )
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
                app.config[
                    "UPLOAD_FOLDER"
                ],
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
# TELEGRAM SEND MESSAGE
# =========================================================

def send_telegram_menu(chat_id):

    keyboard = [
        [
            InlineKeyboardButton(
                "📖 How to Use",
                callback_data="howto"
            ),
            InlineKeyboardButton(
                "🧠 About Model",
                callback_data="model"
            )
        ],
        [
            InlineKeyboardButton(
                "⚠️ Disclaimer",
                callback_data="disclaimer"
            ),
            InlineKeyboardButton(
                "ℹ️ About Project",
                callback_data="about"
            )
        ]
    ]

    reply_markup = InlineKeyboardMarkup(
        keyboard
    )

    telegram_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                "🤖 Skin Disease AI Bot\n\n"
                "Choose an option below:"
            ),
            "reply_markup": json.dumps(
                reply_markup.to_dict()
            )
        }
    )
def send_telegram_message(
    chat_id,
    text
):

    return telegram_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text
        }
    )


# =========================================================
# TELEGRAM PHOTO DOWNLOAD
# =========================================================

def download_telegram_photo(
    file_id,
    destination
):

    result = telegram_api(
        "getFile",
        {
            "file_id": file_id
        }
    )

    file_path = result[
        "result"
    ][
        "file_path"
    ]

    download_url = (
        f"https://api.telegram.org/file/"
        f"bot{TOKEN}/{file_path}"
    )

    with urllib.request.urlopen(
        download_url,
        timeout=120
    ) as response:

        with open(
            destination,
            "wb"
        ) as output:

            output.write(
                response.read()
            )


# =========================================================
# TELEGRAM WEBHOOK
# =========================================================

def handle_button(chat_id, callback_data):

    if callback_data == "howto":

        send_telegram_message(
            chat_id,
            "📖 How to Use\n\n"
            "1️⃣ Send a clear skin image.\n"
            "2️⃣ Wait while the AI analyzes it.\n"
            "3️⃣ The bot will show the predicted class and confidence.\n\n"
            "⚠️ Use the result only as an AI prediction."
        )

    elif callback_data == "model":

        send_telegram_message(
            chat_id,
            "🧠 About the Model\n\n"
            "🤖 Model: Convolutional Neural Network (CNN)\n"
            "🖼️ Image Size: 224 × 224 pixels\n"
            "📊 Classes: 10\n\n"
            "The model analyzes the uploaded image and "
            "returns its predicted class."
        )

    elif callback_data == "disclaimer":

        send_telegram_message(
            chat_id,
            "⚠️ Disclaimer\n\n"
            "This bot provides an AI model prediction only.\n\n"
            "It is NOT a medical diagnosis and should not "
            "be used as a substitute for professional medical advice."
        )

    elif callback_data == "about":

        send_telegram_message(
            chat_id,
            "ℹ️ About Project\n\n"
            "🩺 Skin Disease Detection AI\n\n"
            "This project uses a CNN-based deep learning "
            "model to classify skin images into 10 categories.\n\n"
            "🤖 Built as an AI/ML project."
        )
@app.route(
    "/telegram-webhook",
    methods=["POST"]
)
def telegram_webhook():

    if not TOKEN:

        return (
            "Telegram bot is not configured",
            503
        )

    try:

        data = request.get_json(
            force=True
        )

        print(
            "Telegram update received."
        )

        # =================================================
        # INLINE BUTTON CALLBACK
        # =================================================

        callback_query = data.get(
            "callback_query"
        )

        if callback_query:

            callback_chat_id = callback_query.get(
                "message",
                {}
            ).get(
                "chat",
                {}
            ).get(
                "id"
            )

            callback_data = callback_query.get(
                "data"
            )

            if callback_chat_id and callback_data:

                handle_button(
                    callback_chat_id,
                    callback_data
                )

            telegram_api(
                "answerCallbackQuery",
                {
                    "callback_query_id": callback_query["id"]
                }
            )

            return "OK"

        message = data.get(
            "message"
        )

        if not message:

            return "OK"

        chat = message.get(
            "chat",
            {}
        )

        chat_id = chat.get(
            "id"
        )

        if not chat_id:

            return "OK"

        # =================================================
        # /start
        # =================================================

        text = message.get(
            "text",
            ""
        )

        if text == "/start":

            send_telegram_menu(chat_id)

            return "OK"

        # =================================================
        # /help
        # =================================================

        if text == "/help":

            send_telegram_message(
                chat_id,
                "📌 Available Commands:\n\n"
                "/start - Start the bot\n"
                "/help - Show available commands\n"
                "/about - About the project\n\n"
                "📸 Send a skin image to get a prediction."
            )

            return "OK"

        # =================================================
        # /about
        # =================================================

        if text == "/about":

            send_telegram_message(
                chat_id,
                "🩺 Skin Disease Detection Bot\n\n"
                "🤖 CNN-based deep learning project "
                "for skin image classification.\n\n"
                "🧠 Model: CNN\n"
                "🖼️ Image Size: 224 × 224\n"
                "📊 Classes: 10\n\n"
                "⚠️ This is an AI model prediction, "
                "not a medical diagnosis."
            )

            return "OK"

        # =================================================
        # PHOTO
        # =================================================

        if "photo" in message:

            send_telegram_message(
                chat_id,
                "🔍 Analyzing the image... Please wait."
            )

            photo_list = message[
                "photo"
            ]

            photo = photo_list[
                -1
            ]

            file_id = photo[
                "file_id"
            ]

            image_path = os.path.join(
                UPLOAD_FOLDER,
                "telegram_image.jpg"
            )

            download_telegram_photo(
                file_id,
                image_path
            )

            disease, confidence = (
                predict_image(
                    image_path
                )
            )

            send_telegram_message(
                chat_id,
                (
                    "🔍 Prediction Result\n\n"
                    f"🦠 Disease: {disease}\n"
                    f"📊 Confidence: {confidence:.2f}%\n\n"
                    "⚠️ This is an AI model prediction, "
                    "not a medical diagnosis."
                )
            )

            return "OK"

        return "OK"

    except Exception as e:

        print(
            "Telegram Webhook Error:",
            repr(e)
        )

        try:

            if chat_id:

                send_telegram_message(
                    chat_id,
                    "❌ Sorry, I couldn't process this request."
                )

        except Exception as send_error:

            print(
                "Telegram error message failed:",
                repr(send_error)
            )

        return (
            "Webhook error",
            500
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health"
)
def health():

    return (
        "Skin Disease AI is running!"
    )


# =========================================================
# SET WEBHOOK WHEN APP STARTS
# =========================================================

if TOKEN:

    setup_telegram_webhook()


# =========================================================
# RUN APPLICATION LOCALLY
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
