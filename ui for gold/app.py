from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained model (ensure the model is in the same directory)
model = joblib.load('gold_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        spx = float(data['spx'])
        uso = float(data['uso'])
        slv = float(data['slv'])
        eurusd = float(data['eurusd'])
        gld_lag1 = float(data['gld_lag1'])
        gld_lag2 = float(data['gld_lag2'])
        gld_lag3 = float(data['gld_lag3'])

        # Prepare the data for prediction
        input_data = np.array([[spx, uso, slv, eurusd, gld_lag1, gld_lag2, gld_lag3]])

        # Make the prediction using the loaded model
        predicted_price = model.predict(input_data)[0]

        # Return the result as JSON
        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
