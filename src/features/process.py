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


def stats(df: pd.DataFrame) -> tuple[
    ndarray | int | float | complex, ndarray | int | float | complex, ndarray, ndarray]:
    arr = df.values
    return np.amin(arr), np.amax(arr), np.mean(arr), np.std(arr)


def shape_features(df: pd.DataFrame) -> tuple[ndarray, ndarray | Iterable | int | float]:
    arr = df.values
    return skew(arr), kurtosis(arr)


def rms(x):
    return np.sqrt(np.mean(np.square(np.ediff1d(x))))

if __name__ == "__main__":

    data_folder = Path("../../data/interim/Stress_dataset_labeled/")
    csv_files = list(data_folder.glob("*_labeled.csv"))

    for input_file in csv_files:
        NURSE_ID = input_file.stem[:2]  # Extracts the NURSE_ID by removing the '_labeled' suffix
        print(f"Processing data for NURSE_ID: {NURSE_ID}")
        INPUT_PATH = f"../../data/interim/Stress_dataset_labeled/{NURSE_ID}_labeled.csv"


        df = pd.read_csv(INPUT_PATH)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["label"] = df["label"].astype(int)
        df_raw = df.copy()
        df = df_raw[(np.abs(zscore(df_raw.drop(["datetime", "label", "id"], axis=1))) < 3).all(axis=1)]
        df.reset_index(inplace=True, drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["label"] = df["label"].astype(int)
        df_grouped_date = df.drop(["id"], axis=1).groupby(df.datetime.dt.date)

        cols_to_process = ["EDA", "HR", "TEMP"]
        cols_to_create = [
            "EDA_Min", "EDA_Max", "EDA_Mean", "EDA_Std", "EDA_Skew", "EDA_Kurtosis", "EDA_Num_Peaks", "EDA_Amphitude",
            "EDA_Duration", "HR_Min", "HR_Max", "HR_Mean", "HR_Std", "HR_RMS", "TEMP_Min", "TEMP_Max", "TEMP_Mean",
            "TEMP_Std", "TEMP_RMS", "label"
        ]

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

        features_to_shift = ["EDA_Mean", "HR_Mean", "TEMP_Mean"]
        cols = [f"{x}_{i}" for x in features_to_shift for i in range(10, 0, -1)]
        df_lag_features = pd.DataFrame(columns=cols)

        df_lag_features = pd.concat(
            [df_features[x].shift(i) for i in range(10, 0, -1) for x in features_to_shift], axis=1)
        df_lag_features.columns = cols
        df_lag_features = df_lag_features.dropna()

        df_processed = pd.concat([df_lag_features, df_features.iloc[10:]], axis=1)
        df_processed.reset_index(drop=True, inplace=True)

        df_processed.to_csv(f"../../data/processed/{NURSE_ID}.csv")

    # concat all the processed files
    df = pd.DataFrame()
    for input_file in csv_files:
        NURSE_ID = input_file.stem[:-8]
        df_temp = pd.read_csv(f"../../data/processed/{NURSE_ID}.csv")
        df = pd.concat([df, df_temp], axis=0)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("../../data/processed/combined.csv")