from flask import Flask, render_template, request
import pandas as pd
import joblib
import requests

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('linear_regression_model.pkl')

# Read weather data from CSV file
weather_data = pd.read_csv('weather.csv')

api_key = '9cc5758bdd26da83f9ecc45040d78ca0'

def get_current_weather(city):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        current_temperature = data['main']['temp']
        current_condition = data['weather'][0]['description']
        return current_temperature, current_condition
    else:
        # Handle the case where the request was not successful
        print(f"Failed to retrieve weather data for {city}. Status code: {response.status_code}")
        return None, None

def predict_temperature(city_weather_data):
    # Extract relevant features from the city_weather_data
    features = city_weather_data[['precip_mm', 'wind_kph', 'humidity', 'air_quality_PM2.5']]
    # Use the trained model to predict temperature
    predicted_temperature = model.predict(features)
    return predicted_temperature[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    city = None
    weather_report = None
    if request.method == 'POST':
        city = request.form['city']
        city_weather = weather_data[weather_data['location_name'] == city]
        if not city_weather.empty:
            current_temperature, current_condition = get_current_weather(city)
            predicted_temperature = predict_temperature(city_weather)
            historical_data = city_weather.tail(3).to_dict(orient='records')
            weather_report = {
                'today': (current_temperature, current_condition),
                'predicted_temperature':predicted_temperature,
                'historical_data': historical_data
            }
    return render_template('index.html', city=city, weather_report=weather_report)

if __name__ == '__main__':
    app.run(debug=True)
