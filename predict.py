# Imports the necessary libraries
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.inference import predict_stress_level, load_model, calculate_accuracy, scale_data
from src.features.process import read_and_preprocess_data, create_feature_columns, extract_features, create_lag_features

if __name__ == "__main__":
    # Parses command-line arguments
    parser = argparse.ArgumentParser(description="Stress Detection Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../models",
        help="Path to the directory where the trained models are stored.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/processed/preprocessed_for_inference.csv",
        help="Path to the preprocessed data CSV file.",
    )
    parser.add_argument(
        "--processed",
        type=bool,
        default=False,
        help="Whether the data has been preprocessed or not.",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="../../models/scaler.pkl",
        help="Path to the scaler file.",
    )
    args = parser.parse_args()

    # Sets values of variables based on parsed arguments
    model_path = args.model_path
    data_path = args.data_path
    is_processed = args.processed
    scaler_path = args.scaler_path

    # Loads the XGBoost model
    bst = load_model(model_path)

    if not is_processed:
        # Reads and preprocess the data
        df = read_and_preprocess_data(data_path)

        # Defines the feature columns to process and create
        cols_to_process, cols_to_create = create_feature_columns()

        # Extracts the features from the data and shift them to create lag features
        df_features = extract_features(df, cols_to_process, cols_to_create)
        features_to_shift = ["EDA_Mean", "HR_Mean", "TEMP_Mean"]
        df_lag_features = create_lag_features(df_features, features_to_shift)
        df_processed = pd.concat([df_lag_features, df_features.iloc[10:]], axis=1)
        df_processed.reset_index(drop=True, inplace=True)
    else:
        # Loads preprocessed data
        df_processed = pd.read_csv(data_path)
        # Drops every column that includes "Unnamed" in the name
        df_processed = df_processed.loc[:, ~df_processed.columns.str.contains('^Unnamed')]

    # Extracts the features and target variable
    X = df_processed.drop(["label"], axis=1).values
    y = df_processed["label"].values

    # Scales the features using the scaler file
    X = scale_data(X, scaler_path=scaler_path)

    # Makes predictions using the loaded XGBoost model
    y_pred = predict_stress_level(bst, X)

    # Prints the predicted stress levels
    print("Predicted stress levels:", y_pred)

    # Prints the true stress levels
    print("True stress levels:", y)

    # Prints the accuracy of the predictions
    print("Accuracy:", calculate_accuracy(y, y_pred))

    # Prints the confusion matrix of the predictions
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Prints the classification report of the predictions
    print("Classification Report:")
    print(classification_report(y, y_pred))
