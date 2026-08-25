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
    print("CNN MODEL LOADED SUCCESSFULLY")
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
# GUIDANCE
# =========================================================

def get_guidance(disease, confidence):

    # -----------------------------------------------------
    # LOW CONFIDENCE
    # -----------------------------------------------------

    if confidence < 60:

        guidance_level = (
            "Low AI confidence — the prediction "
            "may not be reliable."
        )

    # -----------------------------------------------------
    # HIGHER CONFIDENCE
    # -----------------------------------------------------

    else:

        guidance_level = (
            "AI prediction with higher model confidence. "
            "This is not a medical diagnosis."
        )


    # -----------------------------------------------------
    # DEFAULT GUIDANCE
    # -----------------------------------------------------

    guidance = [

        "Keep the affected area clean and avoid unnecessary irritation.",

        "Avoid scratching or picking the affected skin.",

        "If the concern persists or changes, consult a qualified healthcare professional."

    ]


    disease_lower = disease.lower()


    # -----------------------------------------------------
    # SERIOUS CONDITIONS
    # -----------------------------------------------------

    if (
        "melanoma" in disease_lower
        or "carcinoma" in disease_lower
    ):

        guidance = [

            "This AI result may indicate a potentially serious skin condition.",

            "Please consult a qualified healthcare professional promptly.",

            "Do not rely on the AI result alone for diagnosis or treatment."

        ]


    elif "eczema" in disease_lower:

        guidance.insert(
            0,
            "Use gentle, fragrance-free skin care products if suitable."
        )


    elif "atopic dermatitis" in disease_lower:

        guidance.insert(
            0,
            "Avoid known skin irritants and use gentle skin care."
        )


    elif "fungal" in disease_lower:

        guidance.insert(
            0,
            "Keep the affected area clean and dry."
        )


    elif "psoriasis" in disease_lower:

        guidance.insert(
            0,
            "Avoid scratching or irritating the affected area."
        )


    return guidance_level, guidance


# =========================================================
# DERMATOLOGIST / HOSPITAL MAP
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
# TRACK MY SKIN
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
                "Send a skin image to get an AI prediction.\n\n"
                "Choose an option below:"
            ),
            "reply_markup": json.dumps(
                reply_markup.to_dict()
            )
        }
    )


# =========================================================
# TELEGRAM SEND MESSAGE
# =========================================================

def send_telegram_message(
    chat_id,
    text,
    reply_markup=None
):

    data = {
        "chat_id": chat_id,
        "text": text
    }

    if reply_markup:

        data["reply_markup"] = json.dumps(
            reply_markup.to_dict()
        )

    return telegram_api(
        "sendMessage",
        data
    )


# =========================================================
# VOICE REPORT
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


        # -------------------------------------------------
        # VOICE MESSAGE
        # -------------------------------------------------

        if confidence < 60:

            voice_text = (
                "Skin Disease AI report. "
                f"The predicted class is {disease}. "
                f"The model confidence is {confidence:.0f} percent. "
                "The confidence is low, so this prediction may not be reliable. "
                "Please send a clear close-up image in good lighting. "
                "This is an AI prediction and not a medical diagnosis."
            )

        else:

            voice_text = (
                "Skin Disease AI report. "
                f"The predicted class is {disease}. "
                f"The model confidence is {confidence:.0f} percent. "
                "This is an AI prediction and not a medical diagnosis. "
                "Please consult a qualified healthcare professional "
                "for medical advice."
            )


        # -------------------------------------------------
        # GENERATE VOICE
        # -------------------------------------------------

        tts = gTTS(
            text=voice_text,
            lang="en"
        )

        tts.save(
            voice_path
        )


        # -------------------------------------------------
        # SEND VOICE TO TELEGRAM
        # -------------------------------------------------

        boundary = (
            "----SkinDiseaseAI"
            + uuid.uuid4().hex
        )

        body = bytearray()


        # chat_id

        body.extend(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                f"{chat_id}\r\n"
            ).encode("utf-8")
        )


        # voice file

        with open(
            voice_path,
            "rb"
        ) as audio:

            audio_data = audio.read()


        body.extend(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; '
                f'name="voice"; filename="{voice_filename}"\r\n'
                f"Content-Type: audio/mpeg\r\n\r\n"
            ).encode("utf-8")
        )

        body.extend(
            audio_data
        )

        body.extend(
            f"\r\n--{boundary}--\r\n".encode(
                "utf-8"
            )
        )


        url = (
            f"https://api.telegram.org/"
            f"bot{TOKEN}/sendVoice"
        )

        req = urllib.request.Request(
            url,
            data=bytes(body),
            method="POST",
            headers={
                "Content-Type":
                    f"multipart/form-data; boundary={boundary}"
            }
        )


        with urllib.request.urlopen(
            req,
            timeout=120
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )


        if not result.get("ok"):

            raise RuntimeError(
                f"Telegram voice API error: {result}"
            )


        print(
            "Voice report sent successfully."
        )


        # Delete temporary voice file

        try:

            os.remove(
                voice_path
            )

        except Exception:
            pass


    except Exception as e:

        print(
            "Voice report error:",
            repr(e)
        )

        send_telegram_message(
            chat_id,
            "🔊 Voice report could not be generated."
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
            "3️⃣ The bot will show the AI prediction and confidence.\n"
            "4️⃣ A voice report is generated when available.\n\n"
            "⚠️ Use the result only as an AI prediction."
        )


    elif callback_data == "model":

        send_telegram_message(
            chat_id,
            "🧠 About the Model\n\n"
            "🤖 Model: Convolutional Neural Network (CNN)\n"
            "🖼️ Image Size: 224 × 224 pixels\n"
            "📊 Classes: 10\n\n"
            "The model analyzes the uploaded image "
            "and returns its predicted class."
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
            "CNN-based deep learning model for skin image classification.\n\n"
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

        text = message.get("text", "").strip()

        if text.startswith("/start"):

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


            # -------------------------------------------------
            # GET PHOTO
            # -------------------------------------------------

            photo_list = message[
                "photo"
            ]

            photo = photo_list[-1]

            file_id = photo[
                "file_id"
            ]


            # -------------------------------------------------
            # SAVE PHOTO
            # -------------------------------------------------

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


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            disease, confidence = (
                predict_image(
                    image_path
                )
            )


            # -------------------------------------------------
            # GUIDANCE
            # -------------------------------------------------

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
            # RESULT
            # IMPORTANT:
            # DISEASE NAME IS SHOWN EVEN IF CONFIDENCE IS LOW
            # =================================================

            disease_lower = disease.lower()


            # -------------------------------------------------
            # SERIOUS CONDITION
            # -------------------------------------------------

            if (
                "melanoma" in disease_lower
                or "carcinoma" in disease_lower
            ):

                hospital_button = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "🏥 Find Nearby Hospital",
                                url=get_dermatologist_url()
                            )
                        ]
                    ]
                )


                # -------------------------------------------------
                # LOW CONFIDENCE SERIOUS RESULT
                # -------------------------------------------------

                if confidence < 60:

                    result_text = (
                        "🔍 Prediction Result\n\n"

                        f"🦠 AI Prediction: {disease}\n"
                        f"📊 Confidence: {confidence:.2f}%\n\n"

                        "⚠️ Low AI confidence — this prediction "
                        "may not be reliable.\n\n"

                        "⚠️ The predicted class may represent "
                        "a potentially serious skin condition.\n\n"

                        "👨‍⚕️ Please consult a qualified "
                        "healthcare professional for proper evaluation.\n\n"

                        "📷 You can also send a clearer, "
                        "close-up image in good lighting.\n\n"

                        "⚠️ This is an AI model prediction, "
                        "not a medical diagnosis."
                    )


                # -------------------------------------------------
                # HIGHER CONFIDENCE SERIOUS RESULT
                # -------------------------------------------------

                else:

                    result_text = (
                        "🔍 Prediction Result\n\n"

                        f"🦠 AI Prediction: {disease}\n"
                        f"📊 Confidence: {confidence:.2f}%\n\n"

                        "⚠️ This result may indicate "
                        "a potentially serious skin condition.\n\n"

                        "👨‍⚕️ Please consult a qualified "
                        "healthcare professional promptly.\n\n"

                        "Do not rely on the AI result alone "
                        "for diagnosis or treatment.\n\n"

                        "⚠️ This is an AI model prediction, "
                        "not a medical diagnosis."
                    )


                reply_markup = hospital_button


            # -------------------------------------------------
            # NORMAL CONDITIONS
            # -------------------------------------------------

            else:

                result_text = (
                    "🔍 Prediction Result\n\n"

                    f"🦠 AI Prediction: {disease}\n"
                    f"📊 Confidence: {confidence:.2f}%\n\n"

                    f"📌 {guidance_level}\n\n"

                    f"{guidance_text}\n\n"

                    "👨‍⚕️ If you are concerned, "
                    "consult a qualified healthcare professional.\n\n"

                    "⚠️ This is an AI model prediction, "
                    "not a medical diagnosis."
                )

                reply_markup = None


            # =================================================
            # SEND RESULT
            # =================================================

            send_telegram_message(
                chat_id,
                result_text,
                reply_markup
            )


            # =================================================
            # VOICE REPORT
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
