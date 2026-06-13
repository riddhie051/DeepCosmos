from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# =========================
# MODEL LOAD
# =========================

model = load_model(
    "model/astronomy_5class_77.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

# =========================
# CLASSES
# =========================

class_names = [
    "Constellation",
    "Galaxy",
    "Nebula",
    "Planet",
    "Star"
]

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================
# HEALTH CHECK
# =========================

@app.route("/healthz")
def healthz():
    return "ok", 200

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        image_path=None
    )

# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            prediction="No Image Selected",
            confidence=None,
            image_path=None
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            prediction="No Image Selected",
            confidence=None,
            image_path=None
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # =========================
    # IMAGE PREPROCESSING
    # =========================

    img = Image.open(filepath)

    img = img.convert("RGB")

    img = img.resize((224, 224))

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img.close()

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # =========================
    # PREDICTION
    # =========================

    preds = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = class_names[
        np.argmax(preds)
    ]

    confidence = round(
        float(np.max(preds)) * 100,
        2
    )

    image_url = url_for(
        "static",
        filename=f"uploads/{file.filename}"
    )

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=image_url
    )

# =========================
# RUN
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )