from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# Load your trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sign_model_small.pkl"
)
model = joblib.load(MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.25


def normalize_landmarks(features):
    coords = np.array(
        features, dtype=np.float32
    ).reshape(21, 3)

    # Make the wrist the origin
    coords = coords - coords[0].copy()

    # Scale normalization
    distances = np.linalg.norm(coords, axis=1)
    scale = np.max(distances)

    if scale < 1e-6:
        return None

    coords = coords / scale

    return coords.flatten()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "landmarks" not in data:
            return jsonify({
                "success": False,
                "error": "Landmarks are missing"
            }), 400

        features = data["landmarks"]

        if len(features) != 63:
            return jsonify({
                "success": False,
                "error": "Expected 63 landmark values"
            }), 400

        if not np.isfinite(
            np.asarray(features, dtype=float)
        ).all():
            return jsonify({
                "success": False,
                "error": "Invalid landmark values"
            }), 400

        normalized = normalize_landmarks(features)

        if normalized is None:
            return jsonify({
        "success": True,
        "prediction": "UNKNOWN",
        "confidence": 0
    })

        # Match the feature names used by your model
        feature_names = []

        for i in range(1, 22):
            feature_names.extend([
                f"x{i}", f"y{i}", f"z{i}"
            ])

        X = pd.DataFrame(
            [normalized],
            columns=feature_names
        )

        probabilities = model.predict_proba(X)[0]
        best_index = int(np.argmax(probabilities))

        confidence = float(probabilities[best_index])
        prediction = str(model.classes_[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "UNKNOWN"

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
})

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
