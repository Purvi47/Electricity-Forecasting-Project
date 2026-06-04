import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('data/electricity.csv')

# Convert datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['Datetime'])
df = df.sort_values('Datetime')

# Time features
df['day'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month
df['hour'] = df['Datetime'].dt.hour
df['weekday'] = df['Datetime'].dt.weekday

# Lag features
df['lag_zone1'] = df['PowerConsumption_Zone1'].shift(1)
df['lag_zone2'] = df['PowerConsumption_Zone2'].shift(1)
df['lag_zone3'] = df['PowerConsumption_Zone3'].shift(1)

df = df.dropna()

# Features
feature_cols = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'day', 'month', 'hour', 'weekday',
    'lag_zone1', 'lag_zone2', 'lag_zone3'
]

X = df[feature_cols]

# Targets
y = df[
    [
        'PowerConsumption_Zone1',
        'PowerConsumption_Zone2',
        'PowerConsumption_Zone3'
    ]
]

# Train-test split (time based)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully")

# Compare actual vs predicted total
sample_count = 10

actual_total = (
    y_test['PowerConsumption_Zone1'].values[:sample_count] +
    y_test['PowerConsumption_Zone2'].values[:sample_count] +
    y_test['PowerConsumption_Zone3'].values[:sample_count]
)

pred_total = (
    y_pred[:sample_count, 0] +
    y_pred[:sample_count, 1] +
    y_pred[:sample_count, 2]
)

print("\nActual vs Predicted Total Consumption")

for i in range(sample_count):
    print("Sample", i + 1, "Actual =", round(actual_total[i], 2), "Predicted =", round(pred_total[i], 2))

# Plot
x = range(sample_count)

plt.figure()

plt.bar([p - 0.2 for p in x], actual_total, width=0.4, label='Actual')
plt.bar([p + 0.2 for p in x], pred_total, width=0.4, label='Predicted')

plt.title("Actual vs Predicted Total Consumption")
plt.xlabel("Samples")
plt.ylabel("Total Consumption")
plt.legend()

plt.show()