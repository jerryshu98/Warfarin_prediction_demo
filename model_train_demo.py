import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import tensorflow as tf

# Reading the Excel file
file_path = './data/PS206767-553247439.xls'
df = pd.read_excel(file_path, sheet_name='Subject Data')

# Selecting required columns
selected_columns = [
    'PharmGKB Subject ID', 'Gender', 'Race (Reported)', 'Race (OMB)',
    'Ethnicity (Reported)', 'Ethnicity (OMB)', 'Age', 'Height (cm)',
    'Weight (kg)', 'Comorbidities', 'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy',
    'Estimated Target INR Range Based on Indication', 'Simvastatin (Zocor)',
    'Valve Replacement', 'Medications', 'Aspirin', 'Amiodarone (Cordarone)',
    'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
    'Therapeutic Dose of Warfarin', 'Indication for Warfarin Treatment'
]

# Selecting and cleaning data
df_selected = df[selected_columns]
df_selected_cleaned = df_selected.dropna(subset=['Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Therapeutic Dose of Warfarin'])

def range_to_average(value):
    if pd.isna(value):
        return np.nan
    elif '-' in value:
        start, end = map(float, value.split('-'))
        return (start + end) / 2
    else:
        return value

df_selected_cleaned['Estimated Target INR'] = df_selected_cleaned['Estimated Target INR Range Based on Indication'].apply(range_to_average)

# Defining feature sets
categorical_columns_sets = [
    # (demographic + gene) - baseline
    ['Gender', 'Race (OMB)', 'Ethnicity (OMB)', 'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'],
    # (demographic + gene)
    ['Gender', 'Race (OMB)', 'Ethnicity (OMB)', 'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Indication for Warfarin Treatment'],
    # (demographic + gene) + other drug
    ['Gender', 'Race (OMB)', 'Ethnicity (OMB)', 'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Indication for Warfarin Treatment'],
    # (demographic + gene) + disease
    ['Gender', 'Race (OMB)', 'Ethnicity (OMB)', 'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Indication for Warfarin Treatment'],
    # (demographic + gene) + other drug + disease
    ['Gender', 'Race (OMB)', 'Ethnicity (OMB)', 'Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Indication for Warfarin Treatment']
]

numerical_columns_sets = [
    ['Height (cm)', 'Weight (kg)'],
    ['Height (cm)', 'Weight (kg)'],
    ['Height (cm)', 'Weight (kg)', 'Simvastatin (Zocor)', 'Aspirin', 'Amiodarone (Cordarone)'],
    ['Height (cm)', 'Weight (kg)', 'Comorbidities', 'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy'],
    ['Height (cm)', 'Weight (kg)', 'Comorbidities', 'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy', 'Simvastatin (Zocor)', 'Aspirin', 'Amiodarone (Cordarone)']
]

def generate_input_data(df, categorical_columns, numerical_columns):
    # One-Hot Encoding categorical data
    one_hot_encoder = OneHotEncoder(sparse=False)
    df_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df[categorical_columns]), columns=one_hot_encoder.get_feature_names_out(categorical_columns))

    # Imputing missing values for numerical data
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Scaling numerical data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

    # Combining categorical and numerical data
    df_final = pd.concat([df_encoded, df_scaled], axis=1)

    X = df_final.values
    y = df['Therapeutic Dose of Warfarin'].values
    return X, y

# Generate input data for all feature sets
input_data_sets = [generate_input_data(df_selected_cleaned, categorical_columns_sets[i], numerical_columns_sets[i]) for i in range(5)]

def run_experiment(X, y, runs=50):
    maes, best_maes, within_20_percent_list, within_20_percent_list_best = [], [], [], []

    for i in range(runs):
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=i)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Define model
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])

        def mean_percentage_error(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            percentage_error = tf.abs((y_true - y_pred) / (tf.abs(y_true) + epsilon))
            return tf.reduce_mean(percentage_error * 100)

        # Compile model
        model.compile(optimizer='adam', loss=mean_percentage_error, metrics=['mae'])

        # Define callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

        # Train model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, model_checkpoint], verbose=0)

        # Evaluate model on validation set
        test_loss, test_mae = model.evaluate(X_val, y_val, verbose=0)
        y_pred = model.predict(X_test)
        maes.append(test_mae)
        within_20 = np.mean((y_test * 0.8 <= y_pred.flatten()) & (y_pred.flatten() <= y_test * 1.2))
        within_20_percent_list.append(within_20)

        # Evaluate best model
        best_model = load_model('best_model.h5', custom_objects={'mean_percentage_error': mean_percentage_error})
        y_pred_best = best_model.predict(X_test)
        within_20_best = np.mean((y_test * 0.8 <= y_pred_best.flatten()) & (y_pred_best.flatten() <= y_test * 1.2))
        within_20_percent_list_best.append(within_20_best)
        test_loss_best, test_mae_best = best_model.evaluate(X_test, y_test, verbose=0)
        best_maes.append(test_mae_best)

    return (
        np.mean(maes), np.std(maes), sum(within_20_percent_list) / len(within_20_percent_list),
        np.mean(best_maes), np.std(best_maes), sum(within_20_percent_list_best) / len(within_20_percent_list_best)
    )

# Run experiments for all feature sets
results = {
    "MAE": [], "STD": [], "20%": [], "Best MAE": [], "Best STD": [], "Best 20%": []
}

for X, y in input_data_sets:
    mean, std, per_20, best_mean, best_std, best_per_20 = run_experiment(X, y)
    results["MAE"].append(mean)
    results["STD"].append(std)
    results["20%"].append(per_20)
    results["Best MAE"].append(best_mean)
    results["Best STD"].append(best_std)
    results["Best 20%"].append(best_per_20)

# Create and save results DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv('experiment_results.csv', index=False)
