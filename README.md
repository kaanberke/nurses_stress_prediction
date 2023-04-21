# Stress Detection Model

This is a machine learning model for stress detection based on physiological signals such as electrodermal activity (EDA), heart rate (HR), and temperature. The model uses XGBoost algorithm to classify the stress level into one of three categories: Normal, Stressed, and Very Stressed.

## Requirements

- Python 3.8 or higher

## Installation

To install the required packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for this project is available in the following link: https://www.nature.com/articles/s41597-022-01361-y

### Dataset Sample

combined_sample.csv

| EDA_Mean_10   | EDA_Mean_9  | EDA_Mean_8 | EDA_Mean_7    | EDA_Mean_6  | EDA_Mean_5 | EDA_Mean_4         | EDA_Mean_3         | EDA_Mean_2         | EDA_Mean_1   | HR_Mean_10        | HR_Mean_9  | HR_Mean_8           | HR_Mean_7   | HR_Mean_6 | HR_Mean_5          | HR_Mean_4   | HR_Mean_3 | HR_Mean_2     | HR_Mean_1         | TEMP_Mean_10 | TEMP_Mean_9         | TEMP_Mean_8 | TEMP_Mean_7 | TEMP_Mean_6         | TEMP_Mean_5       | TEMP_Mean_4        | TEMP_Mean_3         | TEMP_Mean_2        | TEMP_Mean_1 | EDA_Min   | EDA_Max             | EDA_Mean           | EDA_Std               | EDA_Skew            | EDA_Kurtosis        | EDA_Num_Peaks | EDA_Amphitude         | EDA_Duration      | HR_Min | HR_Max | HR_Mean           | HR_Std              | HR_RMS               | TEMP_Min | TEMP_Max | TEMP_Mean  | TEMP_Std            | TEMP_RMS              | label |
|---------------|-------------|------------|---------------|-------------|------------|--------------------|--------------------|--------------------|--------------|-------------------|------------|---------------------|-------------|-----------|--------------------|-------------|-----------|---------------|-------------------|--------------|---------------------|-------------|-------------|---------------------|-------------------|--------------------|---------------------|--------------------|-------------|-----------|---------------------|--------------------|-----------------------|---------------------|---------------------|---------------|-----------------------|-------------------|--------|--------|-------------------|---------------------|----------------------|----------|----------|------------|---------------------|-----------------------|-------|
 | 0.3788545     | 91.131875   | 30.49      | 0.378214      | 90.996875   | 30.49      | 0.378454125        | 90.831875          | 30.49              | 0.3762919375 | 90.65437500000002 | 30.49      | 0.3741300625        | 90.469375   | 30.49     | 0.3744503125       | 90.296875   | 30.49     | 0.37501075    | 90.13187499999998 | 30.49        | 0.374530375         | 90.00125    | 30.49       | 0.373089            | 89.89125          | 30.49              | 0.372288375         | 89.796875          | 30.49       | 0.369246  | 0.375652            | 0.3734896250000001 | 0.0025104551598016897 | -0.9889465676249511 | -0.7802040942830937 | 0.0           | 0.0                   | 0.0               | 89.68  | 89.85  | 89.71187500000002 | 0.06635310373297809 | 0.030532901344549487 | 30.49    | 30.49    | 30.49      | 0.0                 | 0.0                   | 0.0   |
 | 30.489477     | 104.659375  | 34.5       | 30.489477     | 104.764375  | 34.5       | 30.489477          | 104.87125          | 34.5               | 30.489477    | 104.98125         | 34.5       | 30.489477           | 105.063125  | 34.5      | 30.371010875000003 | 105.098125  | 34.5      | 30.272475625  | 105.13125         | 34.5         | 30.3087355          | 105.16125   | 34.5        | 30.340352875        | 105.189375        | 34.5               | 30.176421375        | 105.21437500000002 | 34.5        | 29.893944 | 30.404949           | 29.973828375000004 | 0.12500657271363436   | 2.472215660823218   | 5.9385460672235375  | 0.0           | 0.0                   | 0.0               | 105.23 | 105.23 | 105.23            | 0.0                 | 0.0                  | 34.5     | 34.5     | 34.5       | 0.0                 | 0.0                   | 0.0   |
 | 0.23760525    | 92.425      | 30.17      | 0.23760525    | 92.48       | 30.17      | 0.2379255          | 92.525             | 30.160000000000004 | 0.238566     | 92.57             | 30.15      | 0.23888625          | 92.625      | 30.16     | 0.23792525         | 92.68       | 30.17     | 0.23728475    | 92.805            | 30.17        | 0.2385660000000001  | 92.93       | 30.17       | 0.238566            | 93.05             | 30.160000000000004 | 0.2379255           | 93.17              | 30.15       | 0.230879  | 0.238566            | 0.2363240000000001 | 0.003186874409197827  | -1.065972248058522  | -0.7450043962304314 | 1.0           | 0.0012810000000000046 | 15.58332249251983 | 93.17  | 93.37  | 93.27             | 0.10000000000000142 | 0.03592106040535549  | 30.15    | 30.15    | 30.15      | 0.0                 | 0.0                   | 0.0   |
 | 0.2923755     | 75.4459375  | 28.463125  | 0.29261559375 | 75.485625   | 28.46375   | 0.29101484375      | 75.535625          | 28.44375           | 0.2900544375 | 75.5934375        | 28.433125  | 0.28993434375000005 | 75.6684375  | 28.443125 | 0.2902145625       | 75.735625   | 28.446875 | 0.2901345     | 75.785625         | 28.436875    | 0.28901362500000005 | 75.8528125  | 28.43       | 0.28697203125000004 | 75.95781249999997 | 28.43              | 0.28565100000000004 | 76.0534375         | 28.433125   | 0.28437   | 0.28565100000000004 | 0.28533075         | 0.0005546892711239469 | -1.1547005383791766 | -0.666666666666782  | 0.0           | 0.0                   | 0.0               | 76.03  | 76.18  | 76.1284375        | 0.07124383196986518 | 0.026940795304017256 | 28.43    | 28.45    | 28.443125  | 0.00949917759598146 | 0.0035921060405354212 | 0.0   |
 | 3.18040909375 | 122.9753125 | 32.5234375 | 3.13784225    | 122.5303125 | 32.5084375 | 3.1599464375000004 | 122.09187500000002 | 32.5065625         | 3.20751875   | 121.661875        | 32.5215625 | 3.2343882187500004  | 121.2121875 | 32.53     | 3.2146465625       | 120.7371875 | 32.53     | 3.17115878125 | 120.1965625       | 32.53        | 3.12394675          | 119.5715625 | 32.53       | 3.10772903125       | 118.9421875       | 32.5234375         | 3.112054            | 118.3071875        | 32.5084375  | 3.100962  | 3.22654             | 3.1498956875       | 0.04662358437558554   | 0.6104840764870147  | -1.035568924794963  | 0.0           | 0.0                   | 0.0               | 116.65 | 117.95 | 117.665625        | 0.5374182350599939  | 0.23348689263480685  | 32.5     | 32.53    | 32.5065625 | 0.01240195927061574 | 0.005388159060803451  | 0.0   |

## Usage

### Dataset Preparation

Move "Stress_dataset" folder from the dataset to the ```./data/raw``` folder.
There are 3 steps to follow for dataset preparation. First, run the following command in your terminal:

```bash
python ./src/data/1_combine_data.py
```

This script reads the raw data from the `data/raw` directory, combines them into a single CSV file, and saves the combined data to the `data/interim` directory.
Next run the following command in your terminal:

```bash
python ./src/data/2_merge_data.py
```

This script reads the combined data from the `data/interim` directory, merges them into a single CSV file, and saves the merged data to the `data/interim` directory.
Lastly, run the following command in your terminal:

```bash
python ./src/data/3_label_data.py
```

This script reads the merged data from the `data/interim` directory, labels them into a single CSV file, and saves the labeled data to the `data/processed` directory.

### Training the Model

To train the model, run the `train.py` script with the following arguments:

```bash
python train.py --data_csv [PATH_TO_DATA_CSV] --models_dir [PATH_TO_MODELS_DIRECTORY] --n_splits [NUMBER_OF_SPLITS] --num_boost_round [NUMBER_OF_BOOSTING_ROUNDS] --early_stopping_rounds [NUMBER_OF_EARLY_STOPPING_ROUNDS] --random_search_n_iter [NUMBER_OF_ITERATIONS_FOR_RANDOM_SEARCH] --random_search_cv [NUMBER_OF_CROSS_VALIDATION_FOLDS_FOR_RANDOM_SEARCH] --random_search_verbose [RANDOM_SEARCH_VERBOSITY_LEVEL] --seed [RANDOM_SEED]
```

- `data_csv`: path to the input data CSV file.
- `models_dir`: path to the directory where the trained models will be saved.
- `n_splits`: number of folds for K-Fold cross-validation.
- `num_boost_round`: number of boosting rounds for the XGBoost model.
- `early_stopping_rounds`: number of early stopping rounds for the XGBoost model.
- `random_search_n_iter`: number of iterations for the RandomizedSearchCV.
- `random_search_cv`: number of cross-validation folds for RandomizedSearchCV.
- `random_search_verbose`: verbosity level for RandomizedSearchCV.
- `seed`: random seed for reproducibility.

### Making Predictions

To make predictions using the trained model, run the `predict.py` script with the following arguments:

```bash
python predict.py --model_path [PATH_TO_MODEL] --scaler_path [PATH_TO_SCALER] --input_csv [PATH_TO_INPUT_CSV] --output_csv [PATH_TO_OUTPUT_CSV]
```

- `model_path`: path to the trained model file.
- `scaler_path`: path to the scaler file.
- `input_csv`: path to the input data CSV file.
- `processed`: whether the input data is already processed or not.

## Results

After training the model on the provided dataset, the model achieved an average accuracy of 0.9779 and an average F1 score of 0.9776 on 5-fold cross-validation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
