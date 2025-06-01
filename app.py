from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Include the StudentDropoutPredictor class (must match your training code)
class StudentDropoutPredictor:
    def __init__(self, model, scaler, encoders, feature_names):
        self.model = model
        self.scaler = scaler
        self.encoders = encoders
        self.feature_names = feature_names
        self.model_name = model.__class__.__name__
    
    def predict(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        processed_data = data.copy()
        
        # Label encode categorical features
        for col, encoder in self.encoders.items():
            if col in processed_data.columns:
                processed_data[col] = encoder.transform(processed_data[col].astype(str))
        
        # Fill missing features with 0
        for feature in self.feature_names:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        # Reorder columns
        processed_data = processed_data[self.feature_names]
        
        # Scale if LogisticRegression
        if self.model_name == 'LogisticRegression':
            processed_data = self.scaler.transform(processed_data.values)
        else:
            processed_data = processed_data.values
        
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return {
            'prediction': prediction[0],
            'dropout_probability': float(probability[0][1]),
            'retention_probability': float(probability[0][0])
        }

# Load the saved model object
model = joblib.load('models/student_dropout_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        features = model.feature_names
        return render_template('predict.html', features=features)
    
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            prediction_result = model.predict(form_data)

            # Build probabilities dict for template
            prob_dict = {
                'Retention': prediction_result['retention_probability'],
                'Dropout': prediction_result['dropout_probability']
            }
            confidence = max(prob_dict.values())

            result = {
                'prediction': prediction_result['prediction'],
                'probabilities': prob_dict,
                'confidence': confidence,
                'model_used': model.model_name
            }

            return render_template('results.html', result=result)
        except Exception as e:
            return render_template('results.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON payload provided'}), 400
        
        prediction_result = model.predict(data)
        prob_dict = {
            'Retention': prediction_result['retention_probability'],
            'Dropout': prediction_result['dropout_probability']
        }
        confidence = max(prob_dict.values())

        result = {
            'prediction': prediction_result['prediction'],
            'probabilities': prob_dict,
            'confidence': confidence,
            'model_used': model.model_name
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
