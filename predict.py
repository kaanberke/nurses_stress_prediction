import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.inference import predict_stress_level, load_model, calculate_accuracy, scale_data

from src.features.process import read_and_preprocess_data, create_feature_columns, extract_features, create_lag_features


if __name__ == "__main__":
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

    model_path = args.model_path
    data_path = args.data_path
    is_processed = args.processed
    scaler_path = args.scaler_path

    # Load the model
    bst = load_model(model_path)

    if not is_processed:
        # Read and preprocess data
        df = read_and_preprocess_data(data_path)

        cols_to_process, cols_to_create = create_feature_columns()

        df_features = extract_features(df, cols_to_process, cols_to_create)

        features_to_shift = ["EDA_Mean", "HR_Mean", "TEMP_Mean"]

        df_lag_features = create_lag_features(df_features, features_to_shift)

        df_processed = pd.concat([df_lag_features, df_features.iloc[10:]], axis=1)
        df_processed.reset_index(drop=True, inplace=True)
    else:
        df_processed = pd.read_csv(data_path)
        # drop every column includes unnamed
        df_processed = df_processed.loc[:, ~df_processed.columns.str.contains('^Unnamed')]

    X = df_processed.drop(["label"], axis=1).values
    y = df_processed["label"].values

    X = scale_data(X, scaler_path=scaler_path)

    # Make predictions
    y_pred = predict_stress_level(bst, X)

    # Print predictions
    print("Predicted stress levels:", y_pred)

    # Print true labels
    print("True stress levels:", y)

    # Print accuracy
    print("Accuracy:", calculate_accuracy(y, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, y_pred))
