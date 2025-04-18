from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
import os
from gevent.pywsgi import WSGIServer
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
CORS(app)  # Configure this properly for production

# Setup logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Configuration
MODEL_DIR = "model"
MODEL_PATH = Path(MODEL_DIR) / "real_estate_model.pkl"

# Constants
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pkl'}

# Load model
try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

def preprocess_input(input_data):
    """Preprocess input data for prediction"""
    try:
        df = pd.DataFrame([input_data])
        app.logger.info("Input data processed successfully")
        return df
    except Exception as e:
        app.logger.error(f"Input processing failed: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400

        data = request.get_json()

        # Validate required fields
        required_fields = ['Area', 'Bedrooms', 'Bathrooms', 'Country', 'City']
        missing = [field for field in required_fields if field not in data]
        if missing:
            app.logger.warning(f"Missing fields: {missing}")
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing}'
            }), 400

        # Validate data types
        try:
            input_df = preprocess_input(data)
            prediction = float(model.predict(input_df)[0])
            app.logger.info(f"Prediction successful: {prediction}")
            
            return jsonify({
                'prediction': prediction,
                'status': 'success',
                'message': 'Prediction successful'
            })
            
        except ValueError as e:
            app.logger.error(f"Data validation error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Invalid data format: {str(e)}'
            }), 400

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Production configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # For development
    # app.run(host='0.0.0.0', port=5000, debug=True)
    
    # For production
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    app.logger.info("Server started on port 5000")
    http_server.serve_forever()