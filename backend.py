from flask import Flask, jsonify, request, send_from_directory
import joblib
import numpy as np
import os
import google.generativeai as genai

genai.configure(api_key='AIzaSyAPUWBdEabp0A1XkPNSoS6LQZMa6HNB_vI')

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/get_advice": {"origins": "*"}})

rf_model = joblib.load('mlmodel/crop_rf_model.joblib')
label_encoder = joblib.load('mlmodel/crop_label_encoder.joblib')
scaler = joblib.load('mlmodel/feature_scaler.joblib')


def get_growth_advice(crop):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"What are the essential steps and actions that need to be taken by humans for planting and successfully growing {crop}")
        return response.text
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    print("Root route accessed")
    return 'Flask is running'

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json

        # Extract input features from the JSON request
        nitrogen = data.get("nitrogen")
        phosphorus = data.get("phosphorus")
        potassium = data.get("potassium")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        pH_value = data.get("pH_value")
        rainfall = data.get("rainfall")

        # Prepare the input data for prediction
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH_value, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Predict the crop using the model
        prediction = rf_model.predict(input_scaled)
        crop = label_encoder.inverse_transform(prediction)[0]

        # Return only the crop name
        return jsonify({
            'crop': crop
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_advice', methods=['GET'])
def get_advice():
    try:
        crop = request.args.get('crop')
        
        if not crop:
            return jsonify({'error': 'Please provide a crop name in the request parameters'}), 400
        
        advice = get_growth_advice(crop)
        clean_advice = advice.replace('*', '').strip()
        clean_advice_2 = clean_advice.replace('#', '').strip()
        return jsonify({
           
            'growth_advice': clean_advice_2
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)