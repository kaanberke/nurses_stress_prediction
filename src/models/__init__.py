import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from prettytable import PrettyTable
import os


# Function to preprocess the data
def preprocess_data(df, models_dir, seed=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("label", axis=1))
    y = df.label.values.astype(int)

    scaler_filename = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=seed)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    return X_train, X_val, y_train, y_val, class_weights

# Function to print confusion matrix
def print_confusion_matrix(confusion_matrix, labels):
    table = PrettyTable()
    table.field_names = [""] + labels
    for i, row in enumerate(confusion_matrix):
        table.add_row([labels[i]] + list(row))
    print(table)
