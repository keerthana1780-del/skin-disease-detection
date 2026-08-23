import os
import urllib.request
import urllib.parse
import json
import uuid

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

WEBHOOK_URL = f"https://{RAILWAY_DOMAIN}/telegram-webhook"


if TOKEN:
    print("Telegram bot token found.")
    print("Telegram webhook URL:")
    print(WEBHOOK_URL)
else:
    print("WARNING: TELEGRAM_BOT_TOKEN is not set.")


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

        print(result)

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

        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    print("==========================================")
    print("DOWNLOADING CNN MODEL")
    print("==========================================")

    print(HF_MODEL_URL)

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

                    output.write(chunk)

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

        print(
            "CNN MODEL DOWNLOADED SUCCESSFULLY"
        )

    except Exception as e:

        print(
            "MODEL DOWNLOAD ERROR:",
            repr(e)
        )

        if os.path.exists(MODEL_PATH):

            try:
                os.remove(MODEL_PATH)
            except Exception:
                pass

        raise


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

if not os.path.exists(MODEL_PATH):

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

    print(
        "MODEL LOAD ERROR:",
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
# IDEA 1 - GENERAL GUIDANCE
# =========================================================

def get_guidance(disease, confidence):

    if confidence >= 70:
        guidance_level = "Higher AI confidence — professional review is recommended if concerned."
    elif confidence >= 40:
        guidance_level = "Medium AI confidence — consider professional evaluation."
    else:
        guidance_level = "Low AI confidence — the image may be difficult for the model to classify."

    common = [
        "Keep the affected area clean and avoid unnecessary irritation.",
        "Avoid scratching or picking the affected skin.",
        "Monitor the area for changes in appearance or symptoms.",
        "If the concern persists, changes, or worries you, consult a qualified healthcare professional."
    ]

    disease_lower = disease.lower()

    if "eczema" in disease_lower:

        common.insert(
            0,
            "Use gentle, fragrance-free skin care products if they are suitable for you."
        )

    elif "atopic dermatitis" in disease_lower:

        common.insert(
            0,
            "Avoid known skin irritants and use gentle, fragrance-free skin care."
        )

    elif "fungal" in disease_lower:

        common.insert(
            0,
            "Keep the affected area clean and dry."
        )

    elif "psoriasis" in disease_lower:

        common.insert(
            0,
            "Avoid scratching or irritating the affected area."
        )

    elif (
        "melanoma" in disease_lower
        or "carcinoma" in disease_lower
    ):

        common.insert(
            0,
            "Because this AI result concerns a potentially serious skin condition, seek professional medical evaluation rather than relying on the AI result."
        )

    return guidance_level, common


# =========================================================
# DERMATOLOGIST MAP
# =========================================================

def get_dermatologist_url():

    return (
        "https://www.google.com/maps/search/"
        "?api=1&query=dermatologist+near+me"
    )


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

    guidance = None
    guidance_level = None
    dermatologist_url = get_dermatologist_url()

    if request.method == "POST":

        description = request.form.get(
            "description",
            ""
        ).strip()

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

                guidance_level, guidance = (
                    get_guidance(
                        prediction,
                        confidence
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
        description=description,
        guidance=guidance,
        guidance_level=guidance_level,
        dermatologist_url=dermatologist_url
    )


# =========================================================
# IDEA 2 - TRACK MY SKIN
# =========================================================

@app.route(
    "/track",
    methods=["GET", "POST"]
)
def track():

    old_prediction = None
    old_confidence = None

    new_prediction = None
    new_confidence = None

    old_date = ""
    new_date = ""

    if request.method == "POST":

        old_image = request.files.get(
            "old_image"
        )

        new_image = request.files.get(
            "new_image"
        )

        old_date = request.form.get(
            "old_date",
            ""
        )

        new_date = request.form.get(
            "new_date",
            ""
        )

        if (
            old_image
            and old_image.filename
            and new_image
            and new_image.filename
        ):

            old_filename = (
                "old_"
                + str(uuid.uuid4())
                + "_"
                + secure_filename(
                    old_image.filename
                )
            )

            new_filename = (
                "new_"
                + str(uuid.uuid4())
                + "_"
                + secure_filename(
                    new_image.filename
                )
            )

            old_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                old_filename
            )

            new_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                new_filename
            )

            old_image.save(
                old_path
            )

            new_image.save(
                new_path
            )

            try:

                old_prediction, old_confidence = (
                    predict_image(
                        old_path
                    )
                )

                new_prediction, new_confidence = (
                    predict_image(
                        new_path
                    )
                )

            except Exception as e:

                print(
                    "Tracking prediction error:",
                    repr(e)
                )

    return render_template(
        "track.html",
        old_prediction=old_prediction,
        old_confidence=old_confidence,
        new_prediction=new_prediction,
        new_confidence=new_confidence,
        old_date=old_date,
        new_date=new_date
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
                "Send a skin image to get an AI prediction.\n\n"
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
# IDEA 3 - VOICE REPORT
# =========================================================

def send_voice_report(
    chat_id,
    disease,
    confidence
):

    try:

        from gtts import gTTS

        voice_folder = os.path.join(
            UPLOAD_FOLDER,
            "voice"
        )

        os.makedirs(
            voice_folder,
            exist_ok=True
        )

        voice_filename = (
            "report_"
            + str(uuid.uuid4())
            + ".mp3"
        )

        voice_path = os.path.join(
            voice_folder,
            voice_filename
        )

        if confidence >= 70:
            confidence_text = "high"
        elif confidence >= 40:
            confidence_text = "medium"
        else:
            confidence_text = "low"

        voice_text = (
            "Skin Disease AI report. "
            f"The predicted class is {disease}. "
            f"The model confidence is {confidence:.0f} percent. "
            f"The confidence level is {confidence_text}. "
            "This is an AI prediction and not a medical diagnosis. "
            "Please consult a qualified healthcare professional "
            "for medical advice."
        )

        tts = gTTS(
            text=voice_text,
            lang="en"
        )

        tts.save(
            voice_path
        )

        with open(
            voice_path,
            "rb"
        ) as audio:

            telegram_api(
                "sendVoice",
                {
                    "chat_id": chat_id,
                    "voice": audio
                }
            )

    except Exception as e:

        print(
            "Voice report error:",
            repr(e)
        )

        send_telegram_message(
            chat_id,
            "🔊 Voice report could not be generated. "
            "Here is the text report instead."
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
# TELEGRAM BUTTONS
# =========================================================

def handle_button(
    chat_id,
    callback_data
):

    if callback_data == "howto":

        send_telegram_message(
            chat_id,
            "📖 How to Use\n\n"
            "1️⃣ Send a clear skin image.\n"
            "2️⃣ Wait while the AI analyzes it.\n"
            "3️⃣ The bot will show the predicted class and confidence.\n"
            "4️⃣ A voice report is also generated when available.\n\n"
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
            "CNN-based deep learning model for skin image "
            "classification.\n\n"
            "🧠 CNN\n"
            "🖼️ 224 × 224 image size\n"
            "📊 10 classes\n"
            "👨‍⚕️ Healthcare locator\n"
            "📊 Skin tracking\n"
            "🔊 Telegram voice report"
        )


# =========================================================
# TELEGRAM WEBHOOK
# =========================================================

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

    chat_id = None

    try:

        data = request.get_json(
            force=True
        )

        print(
            "Telegram update received."
        )


        # =================================================
        # CALLBACK BUTTON
        # =================================================

        callback_query = data.get(
            "callback_query"
        )

        if callback_query:

            callback_chat_id = (
                callback_query
                .get("message", {})
                .get("chat", {})
                .get("id")
            )

            callback_data = (
                callback_query
                .get("data")
            )

            if (
                callback_chat_id
                and callback_data
            ):

                handle_button(
                    callback_chat_id,
                    callback_data
                )

            telegram_api(
                "answerCallbackQuery",
                {
                    "callback_query_id":
                        callback_query["id"]
                }
            )

            return "OK"


        # =================================================
        # MESSAGE
        # =================================================

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

            send_telegram_menu(
                chat_id
            )

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
                "📸 Send a skin image to get an AI prediction."
            )

            return "OK"


        # =================================================
        # /about
        # =================================================

        if text == "/about":

            send_telegram_message(
                chat_id,
                "🩺 Skin Disease Detection Bot\n\n"
                "🤖 CNN-based deep learning project.\n\n"
                "🧠 Model: CNN\n"
                "🖼️ Image Size: 224 × 224\n"
                "📊 Classes: 10\n"
                "🔊 Voice Report: Available\n\n"
                "⚠️ AI prediction only — not a medical diagnosis."
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

            photo = photo_list[-1]

            file_id = photo[
                "file_id"
            ]

            image_path = os.path.join(
                UPLOAD_FOLDER,
                "telegram_"
                + str(uuid.uuid4())
                + ".jpg"
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


            # =================================================
            # GUIDANCE
            # =================================================

            guidance_level, guidance = (
                get_guidance(
                    disease,
                    confidence
                )
            )

            guidance_text = "\n".join(
                [
                    "• " + item
                    for item in guidance[:3]
                ]
            )


            # =================================================
            # TEXT RESULT
            # =================================================

            result_text = (
                "🔍 Prediction Result\n\n"
                f"🦠 Disease: {disease}\n"
                f"📊 Confidence: {confidence:.2f}%\n\n"
                f"📌 {guidance_level}\n\n"
                f"{guidance_text}\n\n"
                "👨‍⚕️ If you are concerned, consult a qualified "
                "healthcare professional.\n\n"
                "⚠️ This is an AI model prediction, "
                "not a medical diagnosis."
            )

            send_telegram_message(
                chat_id,
                result_text
            )


            # =================================================
            # IDEA 3 VOICE REPORT
            # =================================================

            send_voice_report(
                chat_id,
                disease,
                confidence
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
