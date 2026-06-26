from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model once on startup
print("Loading model...")
detector = pipeline("image-classification", model="Organika/sdxl-detector")
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
        results = detector(img)

        # Model returns labels: "artificial" / "real"
        ai_score = next(
            (r["score"] for r in results if r["label"].lower() in ("artificial", "ai")),
            None
        )
        if ai_score is None:
            # fallback: 1 - real score
            real_score = next(
                (r["score"] for r in results if r["label"].lower() == "real"),
                0.5
            )
            ai_score = 1 - real_score

        return jsonify(
            success=True,
            filename=file.filename,
            ai_probability=round(ai_score * 100, 1),
            raw=results
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, message=str(e)), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
