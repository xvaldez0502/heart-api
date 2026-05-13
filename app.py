from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'CardioGuard API is running',
        'endpoint': 'POST /predict with 13 clinical features as JSON'
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([data[f] for f in FEATURES]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = float(model.predict_proba(features_scaled)[0][1])
        result = 'High Risk' if prediction == 1 else 'Low Risk'
        return jsonify({
            'prediction': result,
            'probability': round(probability, 4)
        })
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
