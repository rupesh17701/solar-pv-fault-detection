"""FastAPI service for the solar panel fault-detection CNN.

Serves a single-page upload UI at '/' and a JSON inference endpoint at
'POST /predict'. Loads the Keras model trained by train.py once at
startup and reuses it for every request.
"""
import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "app", "model", "solar_fault_classifier.h5")

app = FastAPI(title="Solar PV Fault Detector")
model = None


@app.on_event("startup")
def _load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run app/train.py first."
        )
    model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html>
    <head>
      <title>Solar PV Fault Detector</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 560px; margin: 3rem auto; padding: 0 1rem; }
        h1 { font-size: 1.4rem; }
        #drop { border: 2px dashed #999; border-radius: 8px; padding: 2.5rem; text-align: center; color: #666; cursor: pointer; }
        #drop.hover { border-color: #2563eb; color: #2563eb; }
        #preview { max-width: 100%; margin-top: 1rem; border-radius: 8px; display: none; }
        #result { margin-top: 1.5rem; font-size: 1.1rem; }
        .clean { color: #16a34a; font-weight: 600; }
        .defect { color: #dc2626; font-weight: 600; }
        input[type=file] { display: none; }
      </style>
    </head>
    <body>
      <h1>Solar Panel Fault Detector</h1>
      <p>Upload a photo of a solar panel to check whether it is clean or shows a fault (dust, bird drop, snow, physical or electrical damage).</p>
      <label id="drop" for="file">Click or drop an image here</label>
      <input id="file" type="file" accept="image/*">
      <img id="preview">
      <div id="result"></div>
      <script>
        const drop = document.getElementById('drop');
        const fileInput = document.getElementById('file');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');

        drop.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
        ['dragover'].forEach(evt => drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.add('hover'); }));
        ['dragleave', 'drop'].forEach(evt => drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.remove('hover'); }));
        drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

        async function handleFile(file) {
          preview.src = URL.createObjectURL(file);
          preview.style.display = 'block';
          result.textContent = 'Predicting...';
          const form = new FormData();
          form.append('file', file);
          try {
            const res = await fetch('/predict', { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'prediction failed');
            const cls = data.label === 'Clean' ? 'clean' : 'defect';
            result.innerHTML = `<span class="${cls}">${data.label}</span> &mdash; confidence ${data.confidence}`;
          } catch (err) {
            result.textContent = 'Error: ' + err.message;
          }
        }
      </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((128, 128))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = float(model.predict(arr, verbose=0)[0][0])
    label = "Defect" if pred > 0.5 else "Clean"
    confidence = round(pred if label == "Defect" else 1 - pred, 4)

    return JSONResponse({"label": label, "raw_score": round(pred, 4), "confidence": confidence})
