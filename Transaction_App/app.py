import sys
import os

# absolute path to Modules folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Modules"))

# add to python path
sys.path.append(MODULES_PATH)

from flask import Flask, render_template
import pandas as pd

from utils.inference import load_model, predict_next_day

app = Flask(__name__)

# ===============================
# LOAD MODEL ON START
# ===============================
model, features, scaler = load_model("model/best_model.pkl")


# ===============================
# HOME ROUTE
# ===============================
@app.route("/")
def home():
    try:
        # load dataset
        df = pd.read_csv("data/new_transaction_data.csv")

        # run prediction
        result = predict_next_day(df, model, scaler, features)

        return render_template("index.html", result=result)

    except Exception as e:
        return f"Error: {str(e)}"


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)