# Krishi Mithra: paddy leaf check

The Disease/Pest CNN tool of the Crop Health Agent, as a single service.

- **Backend:** FastAPI + Pydantic. Loads the trained MobileNetV2 model once, answers `POST /api/leaf/predict`.
- **Frontend:** React web app (Vite), mobile-first and tap-first. Works on a phone browser or a laptop.
- **Model:** `paddy_model/mobilenetv2_paddy_final.keras`, 10 paddy classes.

```
Krishimitra/
  paddy_model/   the trained model, class_names.json, metrics, plots (your training output, unchanged)
  backend/       FastAPI service, tests, scripts
  frontend/      React app
```

## Run it (Windows PowerShell)

Backend, in one window. Use Python 3.10 to 3.12.
```powershell
cd Krishimitra\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Wait for `Loaded keras model: 10 classes, preprocess=none`. Then open http://localhost:8000/health: it should say `"model_loaded": true`. http://localhost:8000/docs lets you try the API with a photo.

Frontend, in a second window:
```powershell
cd Krishimitra\frontend
npm install
npm run dev
```
Open http://localhost:5173. To try it on a phone, connect the phone to the same Wi-Fi and open `http://<your-computer-ip>:5173` (allow Python and Node through the Windows firewall on private networks). "Take a photo" opens the phone camera.

**One service for a demo or deployment:** run `npm run build` in `frontend`, then start the backend as above. FastAPI serves the built app too, at http://localhost:8000.

## Preprocessing: this model scales pixels itself

The model's first layer is `Rescaling(1/127.5, offset -1)` (named `mobilenet_preprocess`). So the server sends **raw 0-255 pixels**. If it also applied MobileNetV2 scaling, the model would normalise twice and give wrong answers. `PREPROCESS=auto` (the default) detects this from the loaded model, and the startup log prints the mode it chose. Leave it on `auto`.

## What the model can and cannot do

From `paddy_model/metrics.json` and the classification report (validation set, 2081 images):

| | |
|---|---|
| Top-1 accuracy | 79.5% |
| Top-3 accuracy | 95.2% |
| Weakest classes (F1) | bacterial leaf blight 0.64, downy mildew 0.70, brown spot 0.75 |
| Most common mix-up | normal leaves predicted as hispa (36 of 339) |

Your training notes say these numbers are slightly optimistic, because the validation set was also used for early stopping. So the model is right about four times in five, and the right answer is almost always in its top three. The app is built around that: it always shows the top three candidates, hedges its wording below the high-confidence threshold, and sends low-confidence scans to KVK / Krishi Bhavan.

## Confidence levels

| Top-1 probability | Level | What the app does |
|---|---|---|
| 0.85 or more | high | Names the disease and shows advice |
| 0.60 to 0.85 | medium | "Likely ...", advice marked as general steps, lists the candidates |
| under 0.60 | low | "Not sure yet", no advice, lists the candidates, sends the farmer to an expert |

The percentages never appear in the app (words only), but the API returns them for the Coordinator. The two thresholds are defaults I chose, not measured values. To set them from data, run the model over photos it did not train on:
```powershell
python -m scripts.evaluate E:\path\to\val_dir      # val_dir\<class_id>\*.jpg
```
It prints accuracy per confidence band. Choose `HIGH_CONFIDENCE` where accuracy is one you would trust a farmer with, and set them in `backend/.env` (see `.env.example`).

## API

`POST /api/leaf/predict`: multipart field `file` (JPG or PNG, up to 8 MB). Example response:
```json
{
  "label": "tungro", "disease": "Tungro", "is_healthy": false,
  "confidence": 0.5447, "confidence_level": "low", "needs_expert": true,
  "message": "The scan is not certain. Retake a clearer photo, or show the plant to your local KVK or Krishi Bhavan.",
  "advice": [],
  "predictions": [
    {"rank": 1, "label": "tungro", "name": "Tungro", "confidence": 0.5447, "is_healthy": false},
    {"rank": 2, "label": "bacterial_leaf_blight", "name": "Bacterial Leaf Blight", "confidence": 0.3594, "is_healthy": false},
    {"rank": 3, "label": "brown_spot", "name": "Brown Spot", "confidence": 0.0865, "is_healthy": false}
  ],
  "model_version": "mobilenetv2_paddy_final (keras)", "inference_ms": 120.0
}
```
Also: `GET /health` and `GET /api/model/info` (classes, thresholds, validation metrics). Errors: 400 unreadable or too small photo, 413 too large, 503 model not loaded.

## Tests
```powershell
cd backend;  python -m tests.test_core                       # 24 tests, no extra installs
cd backend;  pip install -r requirements-dev.txt; pytest     # adds the HTTP tests
cd frontend; npm test                                        # 21 tests, no extra installs
```

## Optional: lighter serving with ONNX
`pip install onnx tf2onnx`, then `python -m scripts.convert_onnx`. It exports the model, checks it against the Keras version, and deletes the file if they disagree. With an `.onnx` file present the server uses it automatically, so production can install `requirements-onnx.txt` (no TensorFlow).

## Still to do
- `backend/app/data/disease_info.json` holds generic cultural-practice advice I wrote as a placeholder. Replace it with the KAU-sourced treatment table from the architecture.
- This service only runs the CNN. The regional-outbreak check, the case history, and the Coordinator are separate parts of the architecture and are not built here.
