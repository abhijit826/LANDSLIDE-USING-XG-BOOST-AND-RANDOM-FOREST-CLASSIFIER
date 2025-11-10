# To install dependencies, create a requirements.txt file and run:
# pip install -r requirements.txt
#
# Required libraries:
# scikit-learn, pandas, numpy, Flask, flask-cors

import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from PIL import Image
import io
import numpy as np

# --- Step 1: Initialize Flask App and Load the Model ---
app = Flask(__name__)
CORS(app)

# --- (Optional) Load an Image Analysis Model ---
# This is a placeholder for loading a real image classification model.
# A real model would be trained using a library like TensorFlow or PyTorch.
# For example, with TensorFlow/Keras:
# from tensorflow.keras.models import load_model
# image_model = load_model('landslide_image_model.h5')
image_model = None
app.logger.info("Image analysis model is a placeholder. A real model needs to be trained.")

model_file_path = 'xgboost_landslide_model.pkl'
loaded_model = None
model_columns = None

# Load the saved model file
try:
    with open(model_file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    app.logger.info(f"Model successfully loaded from {model_file_path}.")
    # The columns used for training are crucial for the API
    model_columns = ['Slope_Angle', 'Rainfall_mm', 'Soil_Saturation', 'Vegetation_Cover',
                     'Earthquake_Activity', 'Proximity_to_Water', 'Soil_Type_Gravel',
                     'Soil_Type_Sand', 'Soil_Type_Silt']
except FileNotFoundError:
    app.logger.error(f"Model file '{model_file_path}' not found. The application cannot start without the model.")
    sys.exit(1)
except Exception as e:
    app.logger.error(f"An unexpected error occurred while loading the model: {e}")
    sys.exit(1)

# --- Step 2: Define the Prediction API Endpoint ---

@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to verify that the server is running."""
    return jsonify({'status': 'ok', 'message': 'Prediction server is running.'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data from the frontend, makes a prediction using the loaded model,
    and returns the result.
    """
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format. Expected a single JSON object.'}), 400

        # --- Input Validation ---
        missing_features = [col for col in model_columns if col not in data]
        if missing_features:
            return jsonify({'error': f'Missing required features: {", ".join(missing_features)}'}), 400

        # Create a DataFrame in the correct order for the model
        input_df = pd.DataFrame([data], columns=model_columns)

        # Convert all columns to numeric, raising an error if not possible
        input_df = input_df.apply(pd.to_numeric, errors='raise')

        # --- Prediction ---
        prediction_result = loaded_model.predict(input_df)[0]
        prediction_proba = loaded_model.predict_proba(input_df)[0][1]

        response = {
            'prediction': int(prediction_result),
            'probability': round(float(prediction_proba), 4)
        }

        return jsonify(response)

    except (ValueError, TypeError) as e:
        # Catches errors from pd.to_numeric if data is not numeric
        return jsonify({'error': f'Invalid data type provided. All input values must be numeric. Details: {e}'}), 400
    except Exception as e:
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred on the server.'}), 500

# --- New Endpoint for Image Prediction ---
@app.route('/predict_image', methods=['POST'])
def predict_image():
    """
    Receives an image file, simulates an analysis, and returns a prediction.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in the "image" field'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file:
        try:
            # Read the image file from the request
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # --- Placeholder for Real Image Model Prediction ---
            # In a real application, you would preprocess the image and feed it
            # to your trained neural network model (the 'image_model' loaded above).
            #
            # Example preprocessing:
            # image = image.resize((224, 224))
            # image_array = np.array(image) / 255.0
            # image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
            #
            # Example prediction:
            # prediction_scores = image_model.predict(image_array)
            # predicted_class_index = np.argmax(prediction_scores, axis=1)[0]
            # confidence = float(np.max(prediction_scores))
            # class_names = ['High Risk', 'Low Risk']
            # predicted_class_name = class_names[predicted_class_index]

            # For now, we'll simulate a prediction based on image brightness
            grayscale_image = image.convert('L')
            brightness = np.mean(np.array(grayscale_image))

            predicted_class_name = "Low Risk" if brightness > 128 else "High Risk"
            confidence = abs(brightness - 128) / 128

            response = {'prediction': predicted_class_name, 'confidence': round(float(confidence), 4)}
            return jsonify(response)

        except Exception as e:
            app.logger.error(f"An error occurred during image prediction: {e}", exc_info=True)
            return jsonify({'error': f'Failed to process image. Error: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# --- Step 3: Run the Flask Server ---

if __name__ == '__main__':
    # Note: debug=True is for development only.
    # In a production environment, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=5000)
