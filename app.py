from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join("model", "breast_cancer_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get values from form
            # Features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            perimeter_mean = float(request.form['perimeter_mean'])
            area_mean = float(request.form['area_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            
            # Prepare input for model
            input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
            
            # Predict
            if model:
                pred = model.predict(input_data)[0]
                # Target: 0 = Malignant, 1 = Benign (Checking sklearn dataset again)
                # In sklearn breast_cancer:
                # 'malignant' is 0, 'benign' is 1.
                # Logic: If 0 -> Malignant. If 1 -> Benign.
                if pred == 0:
                    prediction = "Malignant"
                else:
                    prediction = "Benign"
            else:
                prediction = "Model not loaded"
                
        except ValueError:
            prediction = "Invalid input values"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
