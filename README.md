# CardioGuard Heart Risk API

A Flask REST API that predicts heart disease risk based on 13 clinical features.

## Files in this repo

- `app.py` — Flask application with /predict endpoint
- `requirements.txt` — Python dependencies
- `Procfile` — tells Render how to start the app
- `render.yaml` — Render config with Python 3.11 pinned
- `runtime.txt` — Python version for Render
- `heart_model.pkl` — trained scikit-learn model (upload separately)
- `heart_scaler.pkl` — StandardScaler used during training (upload separately)
- `test_api.py` — test script to verify the API

## Predict endpoint

POST /predict with JSON body:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Response:

```json
{
  "prediction": "High Risk",
  "probability": 0.8341
}
```

## Deploying on Render

1. Push all files including heart_model.pkl and heart_scaler.pkl to GitHub
2. Connect the repo to Render as a Web Service
3. Render will detect render.yaml and use Python 3.11 automatically
4. Build command: pip install -r requirements.txt
5. Start command: gunicorn app:app
