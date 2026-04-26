from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        num_cols = ['age', 'avg_glucose_level', 'bmi']
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        binary_cols = ['hypertension', 'heart_disease']
        
        encoded = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
        
        scaled = scaler.transform(input_df[num_cols])
        scaled_df = pd.DataFrame(scaled, columns=num_cols)
        
        final = pd.concat([scaled_df, input_df[binary_cols], encoded_df], axis=1)
        final = final[feature_names]
        
        prob = float(model.predict_proba(final)[0][1])
        pred = 1 if prob > 0.5 else 0
        
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.6:
            risk = "Moderate"
        else:
            risk = "High"
        
        return jsonify({
            "prediction": pred,
            "probability": round(prob * 100, 2),
            "risk_level": risk
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)