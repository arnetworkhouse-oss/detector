from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import os, io

app = Flask(__name__)
CORS(app)

print("Downloading model...")
model_path = hf_hub_download(
    repo_id="LPX55/detection-model-1-ONNX",
    filename="onnx/model.onnx"
)
print(f"Model at: {model_path}")

session     = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g. [1,3,224,224]
img_size    = input_shape[2] if len(input_shape) == 4 else 224
print(f"Model ready. Input shape: {input_shape}")

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    arr  = arr.transpose(2, 0, 1)           # HWC -> CHW
    return arr[np.newaxis].astype(np.float32)

@app.get("/")
def home():
    return "Detector API running 🚀"

@app.post("/detect")
def detect():
    if "file" not in request.files:
        return jsonify(success=False, message="No file uploaded"), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify(success=False, message="Empty filename"), 400
    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = preprocess(img)
        logits = session.run([output_name], {input_name: tensor})[0][0]

        # 2-class softmax (index 1 = AI)
        if logits.shape[0] >= 2:
            exp     = np.exp(logits - np.max(logits))
            ai_prob = float(exp[1] / exp.sum())
        else:
            ai_prob = float(1 / (1 + np.exp(-logits[0])))

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
