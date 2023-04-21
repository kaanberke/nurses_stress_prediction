from typing import Tuple, Iterable

import pandas as pd
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from scipy.stats import skew, kurtosis, zscore
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.figsize"] = (10, 7.5)
pd.options.mode.chained_assignment = None


def stats(df: pd.DataFrame) -> Tuple[
    ndarray | int | float | complex, ndarray | int | float | complex, ndarray, ndarray]:
    arr = df.values
    return np.amin(arr), np.amax(arr), np.mean(arr), np.std(arr)


def shape_features(df: pd.DataFrame) -> Tuple[ndarray, ndarray | Iterable | int | float]:
    arr = df.values
    return skew(arr), kurtosis(arr)


def rms(x):
    return np.sqrt(np.mean(np.square(np.ediff1d(x))))


def read_and_preprocess_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["label"] = df["label"].astype(int)
    df_raw = df.copy()
    df = df_raw[(np.abs(zscore(df_raw.drop(["datetime", "label", "id"], axis=1))) < 3).all(axis=1)]
    df.reset_index(inplace=True, drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["label"] = df["label"].astype(int)
    return df


def create_feature_columns() -> Tuple:
    cols_to_process = ["EDA", "HR", "TEMP"]
    cols_to_create = [
        "EDA_Min", "EDA_Max", "EDA_Mean", "EDA_Std", "EDA_Skew", "EDA_Kurtosis", "EDA_Num_Peaks", "EDA_Amphitude",
        "EDA_Duration", "HR_Min", "HR_Max", "HR_Mean", "HR_Std", "HR_RMS", "TEMP_Min", "TEMP_Max", "TEMP_Mean",
        "TEMP_Std", "TEMP_RMS", "label"
    ]
    return cols_to_process, cols_to_create


def extract_features(df: pd.DataFrame, cols_to_process: list, cols_to_create: list) -> pd.DataFrame:
    df_signals = df[cols_to_process]
    df_features = pd.DataFrame(columns=cols_to_create)

    for i in tqdm(range(0, df.EDA.size, 16)):
        df_sliced = df.iloc[i: i + 32]
        label = df_sliced.label.mode()[0]

        if df_sliced.size < 32:
            continue

        result = []

        # EDA
        result += list(stats(df_sliced[cols_to_process[0]]))
        result += list(shape_features(df_sliced[cols_to_process[0]]))
        peaks, properties = find_peaks(df_sliced[cols_to_process[0]], width=5)
        result += [len(peaks)]
        result += [np.sum(properties["prominences"])]
        result += [np.sum(properties["widths"])]

        # HR
        result += list(stats(df_sliced[cols_to_process[1]]))
        result += [rms(df_sliced[cols_to_process[1]])]

        # TEMP
        result += list(stats(df_sliced[cols_to_process[2]]))
        result += [rms(df_sliced[cols_to_process[2]])]

        result += [label]

        df_features.loc[i / 16] = result

    return df_features

def create_lag_features(df_features: pd.DataFrame, features_to_shift: list) -> pd.DataFrame:
    cols = [f"{x}_{i}" for x in features_to_shift for i in range(10, 0, -1)]
    df_lag_features = pd.DataFrame(columns=cols)

    df_lag_features = pd.concat(
        [df_features[x].shift(i) for i in range(10, 0, -1) for x in features_to_shift], axis=1)
    df_lag_features.columns = cols
    df_lag_features = df_lag_features.dropna()

    return df_lag_features

def concat_and_save_processed_files(saved_files: list, output_path: str):
    df = pd.DataFrame()
    for input_file in saved_files:
        if input_file == output_path:
            continue
        df_temp = pd.read_csv(input_file)
        df = pd.concat([df, df_temp], axis=0)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path)

if __name__ == "__main__":

    data_folder = Path("../../data/interim/")
    csv_files = list(data_folder.glob("*_labeled.csv"))

    for input_file in csv_files:
        NURSE_ID = input_file.stem[:2]  # Extracts the NURSE_ID by removing the '_labeled' suffix
        print(f"Processing data for NURSE_ID: {NURSE_ID}")
        INPUT_PATH = f"../../data/interim/Stress_dataset_labeled/{NURSE_ID}_labeled.csv"

        df = read_and_preprocess_data(INPUT_PATH)

        cols_to_process, cols_to_create = create_feature_columns()

        df_features = extract_features(df, cols_to_process, cols_to_create)

        features_to_shift = ["EDA_Mean", "HR_Mean", "TEMP_Mean"]

        df_lag_features = create_lag_features(df_features, features_to_shift)

        df_processed = pd.concat([df_lag_features, df_features.iloc[10:]], axis=1)
        df_processed.reset_index(drop=True, inplace=True)

        df_processed.to_csv(f"../../data/processed/{NURSE_ID}.csv")

    saved_files = list(Path("../../data/processed/").glob("*.csv"))
    output_path = "../../data/processed/combined.csv"
    concat_and_save_processed_files(saved_files, output_path)
