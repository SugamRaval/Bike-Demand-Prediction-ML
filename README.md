# Bike Demand Prediction System

## Overview

This project implements a machine learning-based system to predict bike demand. <br>
The system takes into account several input parameters to make predictions. These parameters include:
- Date
- Hour
- Temperature (°C)
- Humidity (%)
- Wind Speed (km/h)
- Visibility (km)
- Solar Radiation (MJ/m²)
- Rainfall (mm)
- Snowfall (cm)
- Seasons
- Holiday
- Functioning Day<br>
The system is built using Flask, providing a web interface for users to input data and receive predictions. It also utilizes a machine learning model to make accurate predictions based on the input parameters

## Features

- Predicts bike demand based on environmental and temporal factors.
- Provides a web interface for easy data input and prediction retrieval.
- Utilizes a machine learning model trained on historical bike demand data.
- Handles user input validation and error handling.

## Architecture

The system consists of the following components:

1. **Inference Class (`app.py`):**
   - Responsible for handling user input, scaling data, making predictions, and displaying results.
   - Contains methods to convert user input into model input and obtain predictions.
   - Uses a trained machine learning model and scaler.

2. **Flask Application (`flask_app.py`):**
   - Implements a web interface using Flask to interact with users.
   - Handles user input via HTML forms and sends requests to the `Inference` class for predictions.
   - Renders prediction results on the web interface.

3. **Model Files:**
   - `xgboost_regressor_r3_0_93_v1.pkl`: Contains the trained machine learning model for bike demand prediction.
   - `StandardScaler.pkl`: Contains the trained scaler used to scale input data.

4. **HTML Template (`index.html`):**
   - Provides the structure for the web interface, including input forms and result display.

5. **Data Files:**
   - `bike_demand_predicted_data.csv`: Stores predicted bike demand data.

## Usage

To use the system:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Flask application (`flask_app.py`).
4. Access the web interface in your browser and input the required data.
5. View the predicted bike demand based on the input parameters.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
## Requirements

The project requires the following dependencies:

- Flask
- Pandas
- NumPy
- scikit-learn
