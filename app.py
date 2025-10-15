#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask App for Natural Gas Price Prediction
(Random Forest & Decision Tree)
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__, static_url_path='/static')


# -----------------------------
# 1. Load and Train Models
# -----------------------------
def train_models():
    try:
        data_path = os.path.join("data", "naturalgas.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        # Load dataset
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        if "Date" not in df.columns or "Price" not in df.columns:
            raise ValueError("CSV must contain 'Date' and 'Price' columns")

        # Convert and clean data
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "Price"])

        # Extract features
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

        X = df[["Year", "Month", "Day"]]
        y = df["Price"]

        # ✅ Match HTML dropdown values "1" and "2"
        models = {
            "1": RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y),
            "2": DecisionTreeRegressor(random_state=42).fit(X, y)
        }

        print(f"✅ Models trained successfully on {len(df)} records.")
        print(f"📅 Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
        return models

    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise


# Train all models once at startup
models = train_models()


# -----------------------------
# 2. Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", predic_text="--.-- Dollars")


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))
        day = int(request.form.get("day"))
        method = request.form.get("method")

        if method not in models:
            return render_template("index.html", predic_text="⚠️ Invalid method selected.")

        # Prepare input for prediction
        X_pred = np.array([[year, month, day]])

        # Predict using chosen model
        model = models[method]
        predicted_price = model.predict(X_pred)[0]

        model_name = "Random Forest" if method == "1" else "Decision Tree"
        output = f"{predicted_price:.2f} Dollars ({model_name})"

        return render_template("index.html", predic_text=output)

    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template("index.html", predic_text=f"❌ Internal Server Error: {e}")


# -----------------------------
# 3. Run the App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
