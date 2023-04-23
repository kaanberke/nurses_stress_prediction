# Imports required libraries
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from prettytable import PrettyTable
import os
from sklearn.model_selection import RandomizedSearchCV

# Sets up argument parser for command-line options
parser = argparse.ArgumentParser(description="Stress Detection Model Training")

# Adds command-line arguments
parser.add_argument(
    "--data_csv",
    type=str,
    default="./data/processed/combined.csv",
    help="Path to the input data CSV file.",
)
parser.add_argument(
    "--models_dir",
    type=str,
    default="./models",
    help="Path to the directory where the trained models will be saved.",
)
parser.add_argument(
    "--n_splits",
    type=int,
    default=5,
    help="Number of folds for K-Fold cross-validation.",
)
parser.add_argument(
    "--num_boost_round",
    type=int,
    default=10000,
    help="Number of boosting rounds for the XGBoost model.",
)
parser.add_argument(
    "--early_stopping_rounds",
    type=int,
    default=50,
    help="Number of early stopping rounds for the XGBoost model.",
)
parser.add_argument(
    "--random_search_n_iter",
    type=int,
    default=20,
    help="Number of iterations for the RandomizedSearchCV.",
)
parser.add_argument(
    "--random_search_cv",
    type=int,
    default=3,
    help="Number of cross-validation folds for RandomizedSearchCV.",
)
parser.add_argument(
    "--random_search_verbose",
    type=int,
    default=2,
    help="Verbosity level for RandomizedSearchCV.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)

# Parses command-line arguments and store their values
args = parser.parse_args()

# Replaces the corresponding values in the script with the parsed arguments
data_csv = args.data_csv
models_dir = args.models_dir
n_splits = args.n_splits
num_boost_round = args.num_boost_round
early_stopping_rounds = args.early_stopping_rounds
random_search_n_iter = args.random_search_n_iter
random_search_cv = args.random_search_cv
random_search_verbose = args.random_search_verbose
seed = args.seed

# Defines stress level labels
LABELS = {
    0: "Normal",
    1: "Stressed",
    2: "Very Stressed"
}

# Defines function for data preprocessing
def preprocess_data(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("label", axis=1))
    y = df.label.values.astype(int)

    scaler_filename = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=seed)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    return X_train, X_val, y_train, y_val, class_weights

# Defines function to print confusion matrix
def print_confusion_matrix(confusion_matrix, labels):
    table = PrettyTable()
    table.field_names = [""] + labels
    for i, row in enumerate(confusion_matrix):
        table.add_row([labels[i]] + list(row))
    print(table)

# Creates models folder if it does not exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Loads and preprocess data
df = pd.read_csv(data_csv).iloc[:, 1:]
X, X_val, y, y_val, class_weights = preprocess_data(df)
del df

# Creates Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

# Defines base parameters for the XGBoost model
params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "seed": seed,
}

# Sets the parameter grid for random search
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 2, 3],
    "gamma": [0, 0.1, 0.2, 0.3],
}

# Creates the XGBoost classifier
xgb_clf = xgb.XGBClassifier(**params)

# Performs random search for hyperparameter tuning
random_search = RandomizedSearchCV(
    xgb_clf,
    param_grid,
    n_iter=random_search_n_iter,
    scoring="f1_weighted",
    n_jobs=-1,
    cv=random_search_cv,
    verbose=random_search_verbose,
    random_state=seed,
)

random_search.fit(X, y)

# Gets the best estimator and print the best parameters
best_xgb_clf = random_search.best_estimator_
print("Best Parameters: ", random_search.best_params_)

# Updates XGBoost parameters with the best parameters
params.update(random_search.best_params_)

# Trains the XGBoost model using k-fold cross-validation
accuracy_scores = []
f1_scores = []
confusion_matrices = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Creates the DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Trains the XGBoost model
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        evals=[
            (dtrain, "train"),
            (dtest, "test")
        ],
        verbose_eval=20,
    )

    # Predict on test data
    y_pred = bst.predict(dtest)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    confusion = confusion_matrix(y_test, y_pred)

    # Saves evaluation metrics
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)

    # Prints evaluation metrics
    print_confusion_matrix(confusion, labels=list(LABELS.values()))
    print(f"{i}. KFold Accuracy: {accuracy:.4f}")
    print(f"{i}. KFold F1 Score: {f1:.4f}")

    # Saves the XGBoost model for each fold
    bst.save_model(f"{models_dir}/xgboost_kfold_{i}.model")

# Calculates average accuracy and F1 score
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)

# Prints average accuracy and F1 score
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Prints average confusion matrix
avg_confusion_matrix = np.mean(confusion_matrices, axis=0).astype(int)
print("Average Confusion Matrix:")
print_confusion_matrix(avg_confusion_matrix, labels=list(LABELS.values()))
