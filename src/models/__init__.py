# Imports the required libraries
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from prettytable import PrettyTable
import os


# Function to preprocess the data
def preprocess_data(df, models_dir, seed=42):
    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Applies StandardScaler to input features, excluding the label column
    X = scaler.fit_transform(df.drop("label", axis=1))

    # Extracts labels from the input dataframe
    y = df.label.values.astype(int)

    # Saves the scaler object as a pickle file in the specified models directory
    scaler_filename = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Splits the dataset into training and validation sets with stratification and using a specified random seed
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=seed)

    # Computes class weights to account for imbalanced data
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    # Returns the training and validation sets along with class weights
    return X_train, X_val, y_train, y_val, class_weights


# Function to print confusion matrix in a human-readable format
def print_confusion_matrix(confusion_matrix, labels):
    # Initializes a PrettyTable object
    table = PrettyTable()

    # Sets the field names for the table using the provided labels
    table.field_names = [""] + labels

    # Fills the table with the confusion matrix data
    for i, row in enumerate(confusion_matrix):
        table.add_row([labels[i]] + list(row))

    # Prints the formatted confusion matrix
    print(table)
