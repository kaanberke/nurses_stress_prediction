import pandas as pd
import os

from tqdm import tqdm

COMBINED_DATA_PATH = "../../data/interim/combined_data"
SAVE_PATH = "../../data/interim/merged_data"

os.makedirs(SAVE_PATH, exist_ok=True)

print("Reading data ...")

acc, eda, hr, temp = None, None, None, None

signals = ["acc", "eda", "hr", "temp"]

def read_parallel(signal):
    df = pd.read_csv(
        os.path.join(COMBINED_DATA_PATH, f"combined_{signal}.csv"),
        dtype={"id": str}
    )
    return [signal, df]


results = []
for signal in tqdm(signals):
    result = read_parallel(signal)
    results.append(result)

for i in results:
    globals()[i[0]] = i[1]

# Merge data
print("Merging Data ...")
ids = eda["id"].unique()
columns=["X", "Y", "Z", "EDA", "HR", "TEMP", "id", "datetime"]

def merge_parallel(id):
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)

    acc_id = acc[acc['id'] == id]
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)

    df = acc_id.merge(eda_id, on="datetime", how="outer")
    df = df.merge(temp_id, on="datetime", how="outer")
    df = df.merge(hr_id, on="datetime", how="outer")

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    return df

results = []
for i in tqdm(ids):
    result = merge_parallel(i)
    results.append(result)

new_df = pd.concat(results, ignore_index=True)

print("Saving data ...")
new_df.to_csv(os.path.join(SAVE_PATH, "merged_data.csv"), index=False)