# Import required libraries
import os
import pandas as pd

# Define constants
DATA_PATH = "../../data/raw/Stress_dataset"
SAVE_PATH = "../../data/interim/combined_data"

if __name__ == "__main__":
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Function to process the raw data
    def process_df(df, file):
        start_timestamp = df.iloc[0, 0]
        sample_rate = df.iloc[1, 0]
        new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
        new_df["id"] = file[-2:]
        new_df["datetime"] = [(start_timestamp + i / sample_rate) for i in range(len(new_df))]
        return new_df

    # Define final column names for each signal
    final_columns = {
        "ACC": ["id", "X", "Y", "Z", "datetime"],
        "EDA": ["id", "EDA", "datetime"],
        "HR": ["id", "HR", "datetime"],
        "TEMP": ["id", "TEMP", "datetime"],
    }

    # Define names for each signal data in the CSV files
    names = {
        "ACC.csv": ["X", "Y", "Z"],
        "EDA.csv": ["EDA"],
        "HR.csv": ["HR"],
        "TEMP.csv": ["TEMP"],
    }

    # List desired signals to process
    desired_signals = ["ACC.csv", "EDA.csv", "HR.csv", "TEMP.csv"]

    # Initialize dataframes for each signal
    acc = pd.DataFrame(columns=final_columns["ACC"])
    eda = pd.DataFrame(columns=final_columns["EDA"])
    hr = pd.DataFrame(columns=final_columns["HR"])
    temp = pd.DataFrame(columns=final_columns["TEMP"])

    # Iterate over the data files
    for file in os.listdir(DATA_PATH):
        # Check if the file is a directory
        if not os.path.isdir(os.path.join(DATA_PATH, file)):
            continue
        print(f"Processing {file}")
        for sub_file in os.listdir(os.path.join(DATA_PATH, file)):
            if not os.path.isdir(os.path.join(DATA_PATH, file, sub_file)):
                continue
            for signal in os.listdir(os.path.join(DATA_PATH, file, sub_file)):
                if signal.endswith(".csv") and signal in desired_signals:
                    df = pd.read_csv(os.path.join(DATA_PATH, file, sub_file, signal), names=names[signal], header=None)
                    if not df.empty:
                        processed_df = process_df(df, file)
                        if signal == "ACC.csv":
                            acc = pd.concat([acc, processed_df])
                        if signal == "EDA.csv":
                            eda = pd.concat([eda, processed_df])
                        if signal == "HR.csv":
                            hr = pd.concat([hr, processed_df])
                        if signal == "TEMP.csv":
                            temp = pd.concat([temp, processed_df])

    # Save the combined data for each signal
    print("Saving Data ...")
    acc.to_csv(os.path.join(SAVE_PATH, "combined_acc.csv"), index=False)
    eda.to_csv(os.path.join(SAVE_PATH, "combined_eda.csv"), index=False)
    hr.to_csv(os.path.join(SAVE_PATH, "combined_hr.csv"), index=False)
    temp.to_csv(os.path.join(SAVE_PATH, "combined_temp.csv"), index=False)
