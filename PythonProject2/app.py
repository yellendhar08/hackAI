import numpy as np
import os
import pickle
from flask import Flask, request,jsonify, render_template
from flask_cors import CORS

# Initialize Flask App
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS for frontend requests

# Load the trained model
model_path = os.path.join(os.getcwd(), "model.pkl")
model = pickle.load(open(model_path, "rb"))


# Home Route - Renders the input form
@flask_app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Validate input keys
        required_keys = ["pH", "Turbidity", "TDS", "Dissolved_Oxygen", "Nitrate", "Lead", "E_Coli"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing value for {key}"}), 400

        # Convert to NumPy array with safe parsing
        try:
            features = np.array([[
                float(data["pH"]),
                float(data["Turbidity"]),
                float(data["TDS"]),
                float(data["Dissolved_Oxygen"]),
                float(data["Nitrate"]),
                float(data["Lead"]),
                float(data["E_Coli"])
            ]])
        except ValueError:
            return jsonify({"error": "Invalid input format. Please send numbers only."}), 400

        # Make prediction
        prediction = model.predict(features)[0]  # Extract first value

        # Return JSON response to frontend
        return jsonify({"predicted_disease": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    flask_app.run(debug=True)