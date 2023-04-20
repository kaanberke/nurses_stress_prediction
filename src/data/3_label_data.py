import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import os

from tqdm import tqdm

# Read Files
print("Reading merged_data ...")
PATH = "../../data/interim/merged_data"

df = pd.read_csv(os.path.join(PATH, "merged_data.csv"), dtype={"id": str})
df["datetime"] = pd.to_datetime(df["datetime"].apply(lambda x: x * (10 ** 9)))

print("Reading SurveyResults ...")
survey_path = "../../data/processed/SurveyResults.xlsx"

survey_df = pd.read_excel(survey_path, usecols=["ID", "Start time", "End time", "date", "Stress level"], dtype={"ID": str})
survey_df["Stress level"].replace("na", np.nan, inplace=True)
survey_df.dropna(inplace=True)

survey_df["Start datetime"] =  pd.to_datetime(survey_df["date"].map(str) + ' ' + survey_df["Start time"].map(str))
survey_df["End datetime"] =  pd.to_datetime(survey_df["date"].map(str) + ' ' + survey_df["End time"].map(str))
survey_df.drop(["Start time", "End time", "date"], axis=1, inplace=True)

# Convert SurveyResults.xlsx to GMT-00:00
print("Converting ...")
daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

survey_df1 = survey_df[survey_df["End datetime"] <= daylight].copy()
survey_df1["Start datetime"] = survey_df1["Start datetime"].apply(lambda x: x + timedelta(hours=5))
survey_df1["End datetime"] = survey_df1["End datetime"].apply(lambda x: x + timedelta(hours=5))

survey_df2 = survey_df.loc[survey_df["End datetime"] > daylight].copy()
survey_df2["Start datetime"] = survey_df2["Start datetime"].apply(lambda x: x + timedelta(hours=6))
survey_df2["End datetime"] = survey_df2["End datetime"].apply(lambda x: x + timedelta(hours=6))

survey_df = pd.concat([survey_df1, survey_df2], ignore_index=True)
# survey_df = survey_df.loc[survey_df['Stress level'] != 1.0]

survey_df.reset_index(drop=True, inplace=True)

# Label Data
print("Labelling ...")
ids = df["id"].unique()

def parallel(id):
    new_df = pd.DataFrame(columns=["X", "Y", "Z", "EDA", "HR", "TEMP", "id", "datetime", "label"])

    sdf = df[df["id"] == id].copy()
    survey_sdf = survey_df[survey_df["ID"] == id].copy()

    for _, survey_row in survey_sdf.iterrows():
        ssdf = sdf[(sdf["datetime"] >= survey_row["Start datetime"]) & (sdf["datetime"] <= survey_row["End datetime"])].copy()

        if not ssdf.empty:
            ssdf["label"] = np.repeat(survey_row["Stress level"], len(ssdf.index))
            new_df = pd.concat([new_df, ssdf], ignore_index=True)
        else:
            print(f"{survey_row['ID']} is missing label {survey_row['Stress level']} at {survey_row['Start datetime']} to {survey_row['End datetime']}")
    return new_df

results = []
for i in tqdm(ids):
    result = parallel(i)
    results.append(result)


new_df = pd.concat(results, ignore_index=True)

print("Saving ...")
new_df.to_csv(os.path.join(PATH, "merged_data_labeled.csv"), index=False)
print("Done")