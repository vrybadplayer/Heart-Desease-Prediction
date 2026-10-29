# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("heart_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'age': int(request.form['age']),
        'sex': int(request.form['sex']),
        'chest_pain_type': int(request.form['chest_pain_type']),
        'resting_bp_s': int(request.form['resting_bp_s']),
        'cholesterol': int(request.form['cholesterol']),
        'fasting_blood_sugar': int(request.form['fasting_blood_sugar']),
        'max_heart_rate': int(request.form['max_heart_rate']),
        'exercise_angina': int(request.form['exercise_angina']),
        'oldpeak': float(request.form['oldpeak']),
        'ST_slope': int(request.form['ST_slope'])
    }

    input_df = pd.DataFrame([input_data])

    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'exercise_angina', 'ST_slope']
    input_df[categorical_cols] = input_df[categorical_cols].astype('category')

    # Preprocess and predict
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1] * 100

    if prediction == 1:
        result = "High Chance of Heart Disease ❌" if probability > 50 else "Low Chance of Heart Disease ✅"
    else:
        result = "Low Chance of Heart Disease ✅" if probability <= 50 else "High Chance of Heart Disease ❌"

    # Return the result and probability to be displayed in the UI
    return render_template('index.html', prediction=result, probability=f"{probability:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
