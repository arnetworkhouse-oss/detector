from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import onnxruntime as ort
import urllib.request
import os
import io

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://huggingface.co/minchul/cvprw2023_ai_image_detector/resolve/main/model.onnx"
MODEL_PATH = "model.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading ONNX model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis].astype(np.float32)

print("Loading model...")
download_model()
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Model ready.")

@app.get("/")
def home():
    return "Detector API running 🚀"

@app.post("/detect")
def detect():
    if "file" not in request.files:
        return jsonify(success=False, message="No file uploaded"), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify(success=False, message="Empty filename"), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = preprocess(img)
        logits = session.run([output_name], {input_name: tensor})[0][0]

        # single logit -> sigmoid, or 2-class softmax
        if logits.shape[0] == 1:
            ai_prob = float(1 / (1 + np.exp(-logits[0])))
        else:
            ai_prob = float(np.exp(logits[1]) / np.sum(np.exp(logits)))

        return jsonify(
            success=True,
            filename=file.filename,
            ai_probability=round(ai_prob * 100, 1)
        )
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, message=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
