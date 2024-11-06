from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the CSV data and extract valid locations
data = pd.read_csv('Hyderbad_House_price.csv')
valid_locations = data['location'].unique()  # Extract unique locations from the dataset

@app.route('/')
def index():
    # Pass the valid locations to the HTML template
    return render_template('index.html', valid_locations=valid_locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    rate_persqft = request.form['rate_persqft']
    area_insqft = request.form['area_insqft']
    bhk = request.form['bhk']
    location = request.form['location']
    building_status = request.form['building_status']

    # Here you would normally call your model to predict the house price
    # For now, we simulate the price calculation based on the inputs.
    try:
        rate_persqft = float(rate_persqft)
        area_insqft = float(area_insqft)
    except ValueError:
        return "Invalid input! Please enter numerical values for rate per sq ft and area."

    predicted_price = rate_persqft * area_insqft  # Simple calculation for demonstration
    
    # Format the predicted price to INR with commas and two decimal places
    predicted_price_in_inr = f"₹{predicted_price:,.2f}"

    # Render the result page with the prediction
    return render_template('result.html', predicted_price=predicted_price_in_inr)

if __name__ == '__main__':
    app.run(debug=True)
