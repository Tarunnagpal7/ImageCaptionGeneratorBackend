import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from keras.models import  load_model
from helper import extract_features, predict_caption

app = Flask(__name__)
CORS(app) 


model = load_model('./model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameters
max_length = 35  # Replace with the actual max_length from your training
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Routes
@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    try:
        # Extract features
        features = extract_features(img_path)

        # Predict caption
        caption = predict_caption(model, features, tokenizer, max_length)

        # Delete the image from the backend
        os.remove(img_path)

        # Return response
        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)