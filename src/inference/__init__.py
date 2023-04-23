import pickle

import numpy as np
import xgboost as xgb

# Function to scale the input data using a pre-trained scaler
def scale_data(X, scaler_path):
    # Loads the pre-trained scaler from the provided path
    scaler = pickle.load(open(scaler_path, "rb"))
    # Transforms and return the input data using the loaded scaler
    return scaler.transform(X)

# Function to load a pre-trained XGBoost model from the provided path
def load_model(model_path):
    # Creates an XGBoost booster object
    bst = xgb.Booster()
    # Loads the pre-trained model into the booster object
    bst.load_model(model_path)

    return bst

# Function to make predictions using the pre-trained XGBoost model
def predict_stress_level(bst, X):
    # Converts input data into the DMatrix format required by XGBoost
    dtest = xgb.DMatrix(X)
    # Makes predictions using the pre-trained XGBoost model
    y_pred = bst.predict(dtest)
    return y_pred

# Function to calculate the mean absolute error (loss) between true and predicted values
def calculate_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Function to calculate the accuracy between true and predicted values
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
