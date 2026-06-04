import pandas as pd
import pickle

# Load model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv('data/electricity.csv')

df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', dayfirst=True)
df = df.sort_values('Datetime')

# Get last row for lag values
last_row = df.iloc[-1]

lag1 = last_row['PowerConsumption_Zone1']
lag2 = last_row['PowerConsumption_Zone2']
lag3 = last_row['PowerConsumption_Zone3']

print("\nElectricity Consumption Prediction System\n")

# Inputs
datetime_input = input("Enter Datetime (DD-MM-YYYY HH:MM): ")

try:
    temp = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    wind = float(input("Enter Wind Speed: "))
    gdf = float(input("Enter General Diffuse Flows: "))
    diff = float(input("Enter Diffuse Flows: "))
except ValueError:
    print("Invalid numeric input")
    exit()

# Create input dataframe
new_data = pd.DataFrame({
    'Datetime': [datetime_input],
    'Temperature': [temp],
    'Humidity': [humidity],
    'WindSpeed': [wind],
    'GeneralDiffuseFlows': [gdf],
    'DiffuseFlows': [diff],
    'lag_zone1': [lag1],
    'lag_zone2': [lag2],
    'lag_zone3': [lag3]
})

# Convert datetime
new_data['Datetime'] = pd.to_datetime(new_data['Datetime'], errors='coerce', dayfirst=True)

if new_data['Datetime'].isna().any():
    print("Invalid datetime format")
    exit()

# Time features
new_data['day'] = new_data['Datetime'].dt.day
new_data['month'] = new_data['Datetime'].dt.month
new_data['hour'] = new_data['Datetime'].dt.hour
new_data['weekday'] = new_data['Datetime'].dt.weekday

# Feature columns
feature_cols = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'day', 'month', 'hour', 'weekday',
    'lag_zone1', 'lag_zone2', 'lag_zone3'
]

X_new = new_data[feature_cols]

# Prediction
prediction = model.predict(X_new)

z1, z2, z3 = prediction[0]
total = z1 + z2 + z3

print("\nPrediction Result:")
print("Zone 1:", round(z1, 2))
print("Zone 2:", round(z2, 2))
print("Zone 3:", round(z3, 2))
print("Total:", round(total, 2))

# Simple threshold logic
avg_total = 50000

if total > avg_total:
    print("High electricity demand expected")
else:
    print("Normal electricity demand")