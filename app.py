#Abdujljalil_Hassan_Qaid_Ali_TP073212
from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

model = load('random_forest_model.joblib') 
airline_encoder = load('airline_encoder.joblib')  
stops_encoder = load('stops_encoder.joblib')  
class_encoder = load('class_encoder.joblib')  
arrival_time_encoder = load('arrival_time_encoder.joblib')
departure_time_encoder = load('departure_time_encoder.joblib')
destination_city_encoder = load('destination_city_encoder.joblib')
flight_encoder = load('flight_encoder.joblib')
source_city_encoder = load('source_city_encoder.joblib')
def preprocess_data(form_data):
    # Convert form data to DataFrame
    data = pd.DataFrame([form_data])

    # Apply label encoding
    data['airline'] = airline_encoder.transform(data['airline'])
    data['stops'] = stops_encoder.transform(data['stops'])
    data['class'] = class_encoder.transform(data['class'])
    data['arrival_time'] = arrival_time_encoder.transform(data['arrival_time'])
    data['departure_time'] = departure_time_encoder.transform(data['departure_time'])
    data['destination_city'] = destination_city_encoder.transform(data['destination_city'])
    data['flight'] = flight_encoder.transform(data['flight'])
    data['source_city'] = source_city_encoder.transform(data['source_city'])
    
    # One-hot encoding for other categorical variables
    data_encoded = pd.get_dummies(data, columns=['destination_city', 'arrival_time', 'departure_time', 'source_city'], drop_first=True)
    
    # IMPORTANT: Ensure the final DataFrame has the same columns in the same order as the model expects
    # This may involve adding missing columns with 0s, and ordering the columns correctly
    expected_columns = [ 'flight', 'class', 'duration', 'days_left',
    'destination_city_1', 'destination_city_2', 'destination_city_3', 'destination_city_4', 'destination_city_5',
    'arrival_time_1', 'arrival_time_2', 'arrival_time_3', 'arrival_time_4', 'arrival_time_5',
    'stops_1', 'stops_2',
    'departure_time_1', 'departure_time_2', 'departure_time_3', 'departure_time_4', 'departure_time_5',
    'airline_1', 'airline_2', 'airline_3', 'airline_4', 'airline_5',
    'source_city_1', 'source_city_2', 'source_city_3', 'source_city_4', 'source_city_5']

    data_final = pd.DataFrame(columns=expected_columns)
    for col in expected_columns:
        if col in data_encoded:
            data_final[col] = data_encoded[col]
        else:
            data_final[col] = 0  # Add missing columns with 0s
    
    return data_final

@app.route('/')
def home():
    return render_template('index.html')  # Render your input form here

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    processed_data = preprocess_data(form_data)
    prediction = model.predict(processed_data)[0]
    return render_template('result.html', prediction=prediction)  # Display prediction result

if __name__ == '__main__':
    app.run(debug=True)
