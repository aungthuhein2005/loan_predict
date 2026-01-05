# Loan Approval Predictor

A simple web app and API that predicts loan approval using a pre-trained ML model. The backend is built with **FastAPI** and exposes a `/predict` endpoint; there's a minimal frontend in `index.html` that can call the API and show results.

---

## 🚀 Features

- FastAPI REST API for loan approval predictions
- Frontend UI (`index.html`) for interactive predictions
- Uses pre-trained model saved in `loan_model_bundle.pkl` (model + scaler)
- CORS enabled for local/front-end development

---

## 🧰 Tech stack

- Python 3.10+
- FastAPI
- Uvicorn (ASGI server)
- numpy, pandas
- joblib (for loading the model bundle)

---

## ⚙️ Setup & install

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install fastapi[all] uvicorn pandas numpy joblib
```

3. Make sure the bundled model file `loan_model_bundle.pkl` is present in the project root. The app expects it to contain:

- `model`: trained classifier with `predict` and `predict_proba`
- `scaler`: fitted scaler with a `transform` method

If you don't have the bundle, place it in the project root or update `app.py` to point to the correct path.

---

## ▶️ Running locally

- Start the API (recommended):

```bash
# Development
uvicorn app:app --reload --port 8000
# Or using the bundled run
python app.py
```

- Open `index.html` in your browser (you can open the file directly or serve it via a static file server). The frontend will call `http://localhost:8000/predict`.

---

## 🔍 API

### GET /
Health check

Response:
```json
{ "message": "Loan Prediction API is running." }
```

### POST /predict
Request: JSON payload with the following fields (all required):

- `num_dependents` (int)
- `education` ("Graduate" | "Not Graduate")
- `self_employed` ("Yes" | "No")
- `annual_income` (float)
- `loan_amount` (float)
- `loan_term` (float)
- `cibil_score` (float)
- `residential_assets_value` (float)
- `commerical_assets_value` (float)
- `luxury_assets_value` (float)
- `bank_assets_value` (float)

Response example:

```json
{
  "loan_status": "Approved",
  "confidence": 92.34
}
```

cURL sample:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"num_dependents": 1, "education": "Graduate", "self_employed":"No", "annual_income": 60000, "loan_amount": 15000, "loan_term": 36, "cibil_score": 720, "residential_assets_value": 50000, "commerical_assets_value": 0, "luxury_assets_value": 0, "bank_assets_value": 10000 }'
```

---

## 💡 Notes & tips

- The preprocessing pipeline in `app.py` must match the pipeline used during training (order of features and scaling matter).
- CORS is permissive for development; lock down `allow_origins` in production.
- If you retrain, export both the `model` and `scaler` as a single bundle (e.g., `joblib.dump({"model": model, "scaler": scaler}, "loan_model_bundle.pkl")`).

---

## 🤝 Contributing

Contributions are welcome. Open issues for bugs or feature requests and submit PRs for fixes.

---

## 📄 License

This project does not include a license by default. Consider adding one (e.g., MIT) in `LICENSE` if you want to permit reuse.

---

## 👤 Author

Created by the AUNG THU HEIN
