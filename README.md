# Electricity Forecasting Project

## Overview

This project predicts electricity consumption across three different zones using Machine Learning. The model analyzes historical electricity usage, weather conditions, and time-based features to forecast future power demand. The system helps in understanding consumption patterns and supports efficient energy planning.

## Features

* Multi-zone electricity consumption prediction
* Weather-based forecasting
* Time-based feature engineering
* Lag feature generation using historical consumption data
* Random Forest Regression model
* Model performance evaluation using MSE, RMSE, MAE, and R² Score
* Actual vs Predicted consumption visualization
* Saved model for future predictions

## Dataset

The dataset contains electricity consumption records along with environmental and temporal attributes such as:

* Datetime
* Temperature
* Humidity
* Wind Speed
* General Diffuse Flows
* Diffuse Flows
* Power Consumption Zone 1
* Power Consumption Zone 2
* Power Consumption Zone 3

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Pickle

## Project Structure

```text
Electricity-Forecasting/
│
├── data/
│   └── electricity.csv
│
├── model/
│   └── model.pkl (generated after training)
│
├── train.py
├── predict.py
├── README.md
└── .gitignore
```

## Model Training

The training script performs the following tasks:

* Loads and preprocesses the dataset
* Converts datetime values into useful time-based features
* Creates lag features from historical consumption data
* Trains a Random Forest Regressor model
* Evaluates model performance
* Saves the trained model for future predictions
* Generates visualization comparing actual and predicted consumption

## Prediction

The prediction script allows users to provide:

* Datetime
* Temperature
* Humidity
* Wind Speed
* General Diffuse Flows
* Diffuse Flows

Using these inputs and historical lag values, the model predicts:

* Zone 1 Consumption
* Zone 2 Consumption
* Zone 3 Consumption
* Total Electricity Consumption

## Evaluation Metrics

The model is evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

## Results

The trained model successfully forecasts electricity consumption across multiple zones by combining weather conditions, temporal patterns, and previous consumption trends. Visualization of actual and predicted values helps assess model performance and forecasting accuracy.

## Note

The trained model file (`model.pkl`) is not included in the repository due to GitHub file size limitations. The model can be generated locally by running the training script.

## Future Improvements

* Hyperparameter tuning for improved accuracy
* Advanced ensemble learning techniques
* Deep Learning-based forecasting models
* Real-time electricity demand prediction
* Web-based deployment for interactive forecasting

## Author

Purvi Vishwakarma
