import requests
import json

URL = "https://heart-api-4z7v.onrender.com/predict"
headers = {"Content-Type": "application/json"}

high_risk = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

low_risk = {
    "age": 35, "sex": 0, "cp": 1, "trestbps": 120,
    "chol": 180, "fbs": 0, "restecg": 0, "thalach": 175,
    "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
}

for label, payload in [("High-risk profile", high_risk), ("Low-risk profile", low_risk)]:
    try:
        r = requests.post(URL, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        print(f"{label}: {r.json()}")
    except Exception as e:
        print(f"{label} FAILED: {e}")
