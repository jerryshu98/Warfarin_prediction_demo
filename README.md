# Warfarin Dose Prediction Experiment

## Overview
This project is focused on predicting the therapeutic dose of Warfarin using various patient demographic, genetic, comorbidity, and medication data. It uses deep learning models to analyze the relationship between these factors and the required Warfarin dose. The experiment aims to improve dose prediction accuracy, thus supporting personalized medicine.

## File Structure
- **`warfarin_dose_prediction.py`**: This Python script contains the code for data preprocessing, model training, and evaluation.
- **`data/PS206767-553247439.xls`**: Input Excel file containing patient data.
- **`experiment_results.csv`**: Output CSV file storing the results of the experiments.

## Dependencies
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras

Install the required dependencies using:
```
pip install -r requirements.txt
```

## Data Processing
- The dataset is read from an Excel file.
- Specific columns are selected, and rows with missing essential values are removed.
- Data preprocessing involves one-hot encoding of categorical features, imputing missing numerical values, and scaling numerical data.

## Feature Sets
The experiment is conducted on different feature sets, including combinations of:
- Demographic and genetic information.
- Comorbidities and medications.

## Model
- A deep neural network is used, consisting of several dense layers and dropout layers for regularization.
- Mean Percentage Error (MPE) is used as the loss function, and Mean Absolute Error (MAE) is used for evaluation.

## Running Experiments
- The script splits data into training, validation, and test sets.
- It runs multiple experiments to evaluate model performance on different feature sets.
- The results are stored in `experiment_results.csv`, including MAE, standard deviation, and percentage of predictions within 20% of the true value.

## Results
The output CSV file (`experiment_results.csv`) includes:
- **MAE**: Mean Absolute Error for the different experiments.
- **STD**: Standard deviation of the MAE.
- **20%**: Percentage of predictions within 20% of the true dose value.
- **Best MAE**, **Best STD**, **Best 20%**: Metrics for the best models obtained during training.

## Usage
1. Update the file path to your data in the script if needed.
2. Run the script using:
```
python warfarin_dose_prediction.py
```
3. The results will be saved in `experiment_results.csv`.

