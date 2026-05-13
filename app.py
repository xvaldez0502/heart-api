from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # The order must match the training features
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], 
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)
        
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)