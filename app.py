import os
import urllib.request
import urllib.parse
import json
import uuid
import tempfile

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
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# TELEGRAM
# =========================================================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

RAILWAY_DOMAIN = os.environ.get(
    "RAILWAY_PUBLIC_DOMAIN",
    "skin-disease-detection-production.up.railway.app"
)

WEBHOOK_URL = f"https://{RAILWAY_DOMAIN}/telegram-webhook"


# =========================================================
# TELEGRAM API
# =========================================================

def telegram_api(method, data=None):

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    url = f"https://api.telegram.org/bot{TOKEN}/{method}"

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
# WEBHOOK
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
# MODEL
# =========================================================

HF_MODEL_URL = (
    "https://huggingface.co/"
    "keerthana1780/skin-disease-cnn/"
    "resolve/main/"
    "models/skin_disease_cnn_v2.keras"
)

MODEL_PATH = "/tmp/skin_disease_cnn_v2.keras"


def download_model():

    print("==========================================")
    print("CHECKING CNN MODEL")
    print("==========================================")

    if os.path.exists(MODEL_PATH):

        size = os.path.getsize(MODEL_PATH)

        print("Existing model found.")
        print("Model size:", size, "bytes")

        if size > 100_000_000:
            return

        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    print("==========================================")
    print("DOWNLOADING CNN MODEL")
    print("==========================================")

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

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("==========================================")
print("CNN MODEL LOADED SUCCESSFULLY")
print("==========================================")


# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_SIZE = (224, 224)


# =========================================================
# CLASSES
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
# UPLOAD
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
# PREDICTION
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
            predictions[0][predicted_index]
        ) * 100
    )

    return disease, confidence


# =========================================================
# IDEA 1 — GENERAL GUIDANCE
# =========================================================

def get_guidance(disease, confidence):

    # This is intentionally general.
    # It does NOT diagnose or prescribe treatment.

    if confidence < 50:

        risk = "⚠️ Low model confidence"

        guidance = [
            "The AI confidence is relatively low.",
            "Do not rely on this prediction alone.",
            "If the skin concern persists, changes, or worries you, consider consulting a dermatologist."
        ]

    elif confidence < 75:

        risk = "🟡 Moderate model confidence"

        guidance = [
            "This is an AI-generated classification, not a medical diagnosis.",
            "Avoid relying on the result to choose medicines or treatment.",
            "If the concern persists or changes, consider consulting a qualified healthcare professional."
        ]

    else:

        risk = "🟠 Higher model confidence"

        guidance = [
            "The model has relatively higher confidence in this classification.",
            "This result is still only an AI prediction and should not be treated as a diagnosis.",
            "Consider professional medical evaluation if the concern persists or changes."
        ]

    return risk, guidance


# =========================================================
# IDEA 1 — DERMATOLOGIST SEARCH
# =========================================================

def get_dermatologist_url():

    return (
        "https://www.google.com/maps/search/"
        "?api=1"
        "&query=dermatologist+near+me"
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

    guidance_level = None
    guidance = None
    dermatologist_url = None

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

            unique_filename = (
                str(uuid.uuid4())
                + "_"
                + filename
            )

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                unique_filename
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

                # IDEA 1
                guidance_level, guidance = (
                    get_guidance(
                        prediction,
                        confidence
                    )
                )

                dermatologist_url = (
                    get_dermatologist_url()
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
        guidance_level=guidance_level,
        guidance=guidance,
        dermatologist_url=dermatologist_url
    )


# =========================================================
# IDEA 2 — TRACK MY SKIN
# =========================================================

@app.route(
    "/track",
    methods=["GET", "POST"]
)
def track_skin():

    first_image = None
    second_image = None

    first_prediction = None
    second_prediction = None

    first_confidence = None
    second_confidence = None

    if request.method == "POST":

        image1 = request.files.get(
            "first_image"
        )

        image2 = request.files.get(
            "second_image"
        )

        if image1 and image1.filename:

            filename1 = secure_filename(
                image1.filename
            )

            path1 = os.path.join(
                UPLOAD_FOLDER,
                "track_1_" + filename1
            )

            image1.save(path1)

            first_image = filename1

            try:

                first_prediction, first_confidence = (
                    predict_image(path1)
                )

            except Exception as e:

                print(
                    "Track image 1 error:",
                    repr(e)
                )

        if image2 and image2.filename:

            filename2 = secure_filename(
                image2.filename
            )

            path2 = os.path.join(
                UPLOAD_FOLDER,
                "track_2_" + filename2
            )

            image2.save(path2)

            second_image = filename2

            try:

                second_prediction, second_confidence = (
                    predict_image(path2)
                )

            except Exception as e:

                print(
                    "Track image 2 error:",
                    repr(e)
                )

    return render_template(
        "track.html",
        first_image=first_image,
        second_image=second_image,
        first_prediction=first_prediction,
        second_prediction=second_prediction,
        first_confidence=first_confidence,
        second_confidence=second_confidence
    )


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

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
# TELEGRAM MENU
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
    ]["file_path"]

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
            "3️⃣ The bot returns its predicted class and confidence.\n\n"
            "⚠️ The result is an AI prediction only."
        )

    elif callback_data == "model":

        send_telegram_message(
            chat_id,
            "🧠 About the Model\n\n"
            "Model: CNN\n"
            "Image Size: 224 × 224\n"
            "Classes: 10\n\n"
            "The model classifies the uploaded image."
        )

    elif callback_data == "disclaimer":

        send_telegram_message(
            chat_id,
            "⚠️ Disclaimer\n\n"
            "This bot provides an AI model prediction only.\n"
            "It is NOT a medical diagnosis.\n"
            "Please consult a qualified healthcare professional for medical concerns."
        )

    elif callback_data == "about":

        send_telegram_message(
            chat_id,
            "ℹ️ About Project\n\n"
            "🩺 Skin Disease Detection AI\n\n"
            "CNN-based deep learning project for skin image classification."
        )


# =========================================================
# IDEA 3 — OPTIONAL VOICE REPORT
# =========================================================

def create_voice_report(text):

    try:

        from gtts import gTTS

        filename = (
            "/tmp/"
            + str(uuid.uuid4())
            + ".mp3"
        )

        speech = gTTS(
            text=text,
            lang="en"
        )

        speech.save(
            filename
        )

        return filename

    except Exception as e:

        print(
            "Voice generation unavailable:",
            repr(e)
        )

        return None


def send_voice_report(
    chat_id,
    disease,
    confidence
):

    voice_text = (
        "Skin Disease AI report. "
        f"The model predicted {disease}. "
        f"The model confidence is {confidence:.1f} percent. "
        "This is an AI prediction and not a medical diagnosis. "
        "Please consult a qualified healthcare professional "
        "if you have concerns."
    )

    audio_path = create_voice_report(
        voice_text
    )

    if not audio_path:

        return False

    try:

        with open(
            audio_path,
            "rb"
        ) as audio_file:

            telegram_api(
                "sendVoice",
                {
                    "chat_id": chat_id,
                    "voice": audio_file
                }
            )

        return True

    except Exception as e:

        print(
            "Voice send error:",
            repr(e)
        )

        return False

    finally:

        try:
            os.remove(audio_path)
        except Exception:
            pass


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
        # CALLBACK
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
                callback_query.get("data")
            )

            if callback_chat_id and callback_data:

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

        chat_id = (
            message
            .get("chat", {})
            .get("id")
        )

        if not chat_id:
            return "OK"

        text = message.get(
            "text",
            ""
        )

        # =================================================
        # START
        # =================================================

        if text == "/start":

            send_telegram_menu(
                chat_id
            )

            return "OK"

        # =================================================
        # HELP
        # =================================================

        if text == "/help":

            send_telegram_message(
                chat_id,
                "📌 Commands:\n\n"
                "/start - Start bot\n"
                "/help - Help\n"
                "/about - About project\n\n"
                "📸 Send a skin image for prediction."
            )

            return "OK"

        # =================================================
        # ABOUT
        # =================================================

        if text == "/about":

            send_telegram_message(
                chat_id,
                "🩺 Skin Disease Detection Bot\n\n"
                "CNN-based deep learning project.\n"
                "Image Size: 224 × 224\n"
                "Classes: 10\n\n"
                "⚠️ AI prediction only."
            )

            return "OK"

        # =================================================
        # PHOTO
        # =================================================

        if "photo" in message:

            send_telegram_message(
                chat_id,
                "🔍 Analyzing the image..."
            )

            photo = message[
                "photo"
            ][-1]

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

            risk, guidance = (
                get_guidance(
                    disease,
                    confidence
                )
            )

            guidance_text = "\n".join(
                [
                    "• " + item
                    for item in guidance
                ]
            )

            send_telegram_message(
                chat_id,
                (
                    "🔍 Prediction Result\n\n"
                    f"🦠 Disease: {disease}\n"
                    f"📊 Confidence: {confidence:.2f}%\n\n"
                    f"{risk}\n\n"
                    "📌 General Guidance:\n"
                    f"{guidance_text}\n\n"
                    "👨‍⚕️ Consider consulting a dermatologist "
                    "if you are concerned.\n\n"
                    "⚠️ This is an AI prediction, "
                    "not a medical diagnosis."
                )
            )

            # IDEA 3
            voice_sent = send_voice_report(
                chat_id,
                disease,
                confidence
            )

            if voice_sent:

                send_telegram_message(
                    chat_id,
                    "🔊 Voice report generated above."
                )

            return "OK"

        return "OK"

    except Exception as e:

        print(
            "Telegram Webhook Error:",
            repr(e)
        )

        if chat_id:

            try:

                send_telegram_message(
                    chat_id,
                    "❌ Sorry, I couldn't process this request."
                )

            except Exception:
                pass

        return (
            "Webhook error",
            500
        )


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    return "Skin Disease AI is running!"


# =========================================================
# START WEBHOOK
# =========================================================

if TOKEN:

    setup_telegram_webhook()


# =========================================================
# LOCAL RUN
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
