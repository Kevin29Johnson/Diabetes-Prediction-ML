from joblib import load
from flask import Flask, request, jsonify, render_template,url_for
import numpy as np

app = Flask(__name__)
app.static_folder = 'static'

# Load the pipeline (example)
pipeline = load('pipeline.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if the request content type is JSON
            if request.is_json:
                data = request.get_json()
            else:
                # If not JSON, assume form data or other format
                data = request.form.to_dict()
            
            # Prepare the input data for prediction
            X = [[data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                  data['SkinThickness'], data['Insulin'], data['BMI'],
                  data['DiabetesPedigreeFunction'], data['Age']]]
            
            # Make prediction using the loaded pipeline
            prediction = pipeline.predict(X)
            output = prediction[0]
            
            return render_template('home.html', prediction_text=f'Diabetes Prediction: {"Positive" if output == 1 else "Negative"}')
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'message': 'Invalid request method'})

if __name__ == '__main__':
    app.run(debug=True)
