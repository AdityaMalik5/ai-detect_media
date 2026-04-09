import io
import os
import pickle
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from utils import ALLOWED_EXTENSIONS, INPUT_SIZE, MAX_FILE_SIZE_MB, allowed_file

MODEL_PATH = os.path.join(".", "tmp_checkpoint", "best_model.h5")
SVM_PATH = os.path.join(".", "tmp_checkpoint", "svm_model.pkl")

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── Model loading ────────────────────────────────────────────────────────────
_backbone = None
_svm = None
_scaler = None
_init_error: Optional[str] = None

try:
    import tensorflow as tf

    # Load EfficientNet backbone for feature extraction
    from tensorflow.keras.applications import EfficientNetB0

    _backbone = EfficientNetB0(
        weights="imagenet",
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        pooling="max",
    )
    # Warm up
    _backbone.predict(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32), verbose=0)

    # Load SVM classifier
    if os.path.isfile(SVM_PATH):
        with open(SVM_PATH, "rb") as f:
            data = pickle.load(f)
        _svm = data["svm"]
        _scaler = data["scaler"]
    else:
        _init_error = f"SVM model not found at {SVM_PATH}"
except Exception as exc:
    _init_error = f"Failed to load model: {exc}"


def _preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes, resize to INPUT_SIZE x INPUT_SIZE, normalise to [0,1]."""
    if not file_bytes:
        raise ValueError("Could not decode image — file is empty")
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt or not an image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # shape (1, 128, 128, 3)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return jsonify(
        {
            "model_loaded": _svm is not None and _backbone is not None,
            "model_path": SVM_PATH,
            "init_error": _init_error,
        }
    )


@app.post("/api/predict")
def api_predict():
    # ── validate presence ────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"ok": False, "error": "No file selected"}), 400

    # ── validate extension ───────────────────────────────────────────────
    if not allowed_file(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"ok": False, "error": f"File type not allowed. Accepted: {exts}"}), 400

    # ── validate size ────────────────────────────────────────────────────
    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"ok": False, "error": f"File exceeds {MAX_FILE_SIZE_MB} MB limit"}), 400
    if len(file_bytes) == 0:
        return jsonify({"ok": False, "error": "Uploaded file is empty"}), 400

    # ── model gate ───────────────────────────────────────────────────────
    if _svm is None or _backbone is None:
        return jsonify({"ok": False, "error": _init_error or "Model not loaded"}), 501

    # ── preprocess & predict ─────────────────────────────────────────────
    try:
        tensor = _preprocess_image(file_bytes)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Extract features with EfficientNet, classify with SVM
    features = _backbone.predict(tensor, verbose=0).flatten().reshape(1, -1)
    features_scaled = _scaler.transform(features)
    pristine_prob = float(_svm.predict_proba(features_scaled)[0][1])
    deepfake_prob = 1.0 - pristine_prob
    label = "Pristine" if pristine_prob >= 0.5 else "Deepfake"

    return jsonify(
        {
            "ok": True,
            "label": label,
            "pristine_prob": round(pristine_prob, 6),
            "deepfake_prob": round(deepfake_prob, 6),
            "meta": {"input_size": INPUT_SIZE},
        }
    )


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    # Run: python app_server.py
    # Open: http://localhost:8000
    app.run(host="0.0.0.0", port=8000, debug=False)
