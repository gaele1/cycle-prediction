#Occupiamoci della parte front-end
import pandas as pd
from flask import Flask, request, jsonify #flask prenderà i dati dell'utente e li inserirà come input nell'algoritmo
import joblib  # Example if you saved the model with joblib
import datetime
import numpy as np

from sklearn.preprocessing import StandardScaler

x = 360
app = Flask(__name__)

# Load CSV and Model
csv_path = 'cycle_data_transformed.csv'
df = pd.read_csv(csv_path)

# Load your machine learning model (assuming it's saved as a .pkl file)
model = joblib.load('cycle_prediction_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')  # Load the scaler
scaler_y = joblib.load('scaler_y.pkl')

# Function to predict next cycle and duration using the model
def predict_next_cycle(data):
    """
    Use the machine learning model to predict days until the next cycle and duration.
    :param data: The historical cycle data (last row) to feed to the model.
    :return: Predictions of days until the next cycle and cycle duration.
    """
    # Select relevant features for prediction (your input features: cycle_day_1, cycle_day_2, etc.)
    features = data[[f'cycle_day_{i}' for i in range(1, x+1)] + ['month']]  # Adjust to match model inputs

    # Make predictions using the model
    predictions = model.predict(scaler_X.transform(features))
    
    # Assuming model predicts [days_until_next_cycle, duration_next_cycle]
    days_until_next_cycle = scaler_y.inverse_transform(np.concatenate((predictions[0], np.zeros_like(predictions[1])), axis=1))[:, 0]
    duration_next_cycle = scaler_y.inverse_transform(np.concatenate((np.zeros_like(predictions[0]), predictions[1]), axis=1))[:, 1]
    
    return days_until_next_cycle, duration_next_cycle

# Function to get the last row from the CSV
def get_last_row():
    return df.tail(1)

# Function to append a new row based on user feedback
def append_new_row(date, cycle_flg):
    last_row = df.tail(1).copy()

    # Shift cycle_day_1, cycle_day_2,...,cycle_day_x
    cycle_days = [f'cycle_day_{i}' for i in range(1, x+1)]  # Adjust to your exact number of cycle_day_x columns
    for i in reversed(range(1, len(cycle_days))):
        last_row[cycle_days[i]] = last_row[cycle_days[i - 1]]
    last_row['cycle_day_1'] = last_row['cycle_flg']

    # Update the last row with the new cycle flag
    last_row['cycle_flg'] = cycle_flg
    last_row['date'] = date
    last_row['day'] = pd.to_datetime(date).day
    last_row['month'] = pd.to_datetime(date).month


    # Reset the oldest cycle_day_x
    last_row[cycle_days[-1]] = 0  # Set the oldest day to 0 (no cycle) or any other default value

    # Stop predictions if cycle_flg is 1
    if cycle_flg == 1:
        last_row['days_until_next_cycle'] = None
        last_row['duration_next_cycle'] = None
    else:
        # Use the last row to make predictions for the new row
        days_until, duration = predict_next_cycle(last_row)
        last_row['days_until_next_cycle'] = days_until
        last_row['duration_next_cycle'] = duration
    
    # Append the new row to the CSV
    df = df.append(last_row, ignore_index=True)
    df.to_csv(csv_path, index=False)

# Endpoint to get the last row and predictions
@app.route('/get_last_row', methods=['GET'])
def get_last_row_data():
    last_row = get_last_row()
    
    # Make predictions if no cycle started today
    if last_row['cycle_flg'].values[0] != 1:
        days_until, duration = predict_next_cycle(last_row)
        last_row['days_until_next_cycle'] = days_until
        last_row['duration_next_cycle'] = duration
    
    return jsonify(last_row.to_dict(orient='records'))

# Endpoint to receive and update feedback from user
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    date = data.get('date')
    cycle = data.get('cycle')  # 1 for cycle, 0 for no cycle
    
    # Append a new row with today's feedback
    append_new_row(date, cycle)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
