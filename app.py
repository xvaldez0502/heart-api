from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # enables cross-origin requests so Lovable, your browser, etc. can call this API

# Load the trained model and scaler that were saved during training
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Feature order has to match what the model was trained on
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


@app.route('/', methods=['GET'])
def home():
    # Root endpoint so the API doesn't return a confusing 404 when visited in a browser
    return jsonify({
        'status': 'CardioGuard API is running',
        'usage': 'POST clinical features to /predict',
        'endpoints': {
            '/predict': 'POST with 13 clinical features as JSON',
            '/health': 'GET to check service health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle the CORS preflight request that browsers send before the real POST
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()

        # Build feature vector in the exact order the model was trained with
        features = [data[f] for f in FEATURE_ORDER]
        features_array = np.array(features).reshape(1, -1)

        # Scale the features using the saved scaler from training
        features_scaled = scaler.transform(features_array)

        # Run prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Return human-readable label that matches what the Lovable frontend expects
        result = 'High Risk' if prediction == 1 else 'Low Risk'

        return jsonify({
            'prediction': result,
            'probability': float(probability),
            'raw_prediction': int(prediction)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
