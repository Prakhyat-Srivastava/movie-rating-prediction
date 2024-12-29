# -----------------------------------------------------
# app.py
# A Flask web app to load xgb_best_model.pkl
# and predict a movie's rating based on user input.
# -----------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from xgboost import XGBRegressor

# Initialize Flask
app = Flask(__name__)

# Load the trained model (ensure the path is correct)
MODEL_PATH = os.path.join("model", "xgb_best_model.pkl")
model = joblib.load(MODEL_PATH)

# Define genre columns from your training script
GENRE_COLS = [
    'Action','Adventure','Animation','Biography','Comedy','Crime','Documentary',
    'Drama','Family','Fantasy','History','Horror','Music','Musical','Mystery',
    'News','Romance','Sci-Fi','Sport','Thriller','Unknown','War','Western'
]

# The same features used in training (order is important)
FEATURES = ['Duration', 'Log_Votes', 'Decade'] + GENRE_COLS

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Renders a form for user input (Duration, Log_Votes, Decade, Genre checkboxes).
    On POST, parses the form, constructs a single-row DataFrame, and predicts rating.
    """
    prediction = None

    if request.method == "POST":
        # 1. Get numeric fields (Duration, Log_Votes, Decade)
        duration_str = request.form.get("duration", "0")
        log_votes_str = request.form.get("log_votes", "0")
        decade_str = request.form.get("decade", "0")

        # Convert to float (or int)
        try:
            duration_val = float(duration_str)
        except ValueError:
            duration_val = 0.0

        try:
            log_votes_val = float(log_votes_str)
        except ValueError:
            log_votes_val = 0.0

        try:
            decade_val = float(decade_str)
        except ValueError:
            decade_val = 0.0

        # 2. Handle genre checkboxes
        # If a genre is checked, we set it to 1, otherwise 0
        genre_data = {}
        for g in GENRE_COLS:
            if request.form.get(f"genre_{g}") == "on":
                genre_data[g] = 1
            else:
                genre_data[g] = 0

        # 3. Construct a single-row input
        input_dict = {
            'Duration': duration_val,
            'Log_Votes': log_votes_val,
            'Decade': decade_val,
        }
        input_dict.update(genre_data)

        # Convert dict to DataFrame with the same columns as training
        df_input = pd.DataFrame([input_dict], columns=FEATURES)

        # 4. Predict using the loaded model
        pred_rating = model.predict(df_input)[0]
        # Round or keep as float
        prediction = round(float(pred_rating), 2)

    # Render index.html, passing the prediction to display if available
    return render_template("index.html", prediction=prediction, genres=GENRE_COLS)

if __name__ == "__main__":
    # Run Flask app
    # Host 0.0.0.0 to allow external access if needed, otherwise "127.0.0.1"
    app.run(host="127.0.0.1", port=5000, debug=True)
