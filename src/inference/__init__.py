import pickle

import numpy as np
import xgboost as xgb

def scale_data(X, scaler_path):
    scaler = pickle.load(open(scaler_path, "rb"))
    return scaler.transform(X)

def load_model(model_path):
    bst = xgb.Booster()
    bst.load_model(model_path)

    return bst

def predict_stress_level(bst, X):
    dtest = xgb.DMatrix(X)
    y_pred = bst.predict(dtest)
    return y_pred

def calculate_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
