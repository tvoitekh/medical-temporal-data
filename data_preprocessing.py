import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import os

DATA_PATH = './'
FEATURES_FILE = os.path.join(DATA_PATH, 'features_for_mortality_prediction.csv')
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed_data')
if not os.path.exists(PREPROCESSED_DATA_PATH):
    os.makedirs(PREPROCESSED_DATA_PATH)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}. Please ensure TODO 2 (feature extraction) was run successfully and generated this file.")
        return None
    return pd.read_csv(file_path)

def preprocess_data(df):
    if df is None:
        return None, None, None, None, None, None, None

    print("\nStarting data preprocessing...")

    print("Step 1: Defining target and features...")
    if 'mortality' not in df.columns:
        print("ERROR: Target column 'mortality' not found in the features DataFrame.")
        print("Ensure 'mortality' was correctly included and named during cohort selection and feature merging.")
        return None, None, None, None, None, None, None

    y = df['mortality']

    potential_cols_to_drop = [
        'subject_id', 'hadm_id', 'stay_id',
        'gender',
        'intime', 'outtime', 'deathtime',
        'los', 'los_hours',
        'hospital_expire_flag',
        'first_careunit', 'last_careunit',
        'anchor_year', 'admittime_year', 'admittime'
    ]

    actual_cols_to_drop = [col for col in potential_cols_to_drop if col in df.columns]
    actual_cols_to_drop.append('mortality') # Add target to the list of columns to remove from X

    X = df.drop(columns=actual_cols_to_drop, errors='ignore')
    print(f"Initial number of features before detailed preprocessing: {X.shape[1]}")
    if X.empty:
        print("ERROR: Feature set X is empty after initial drops. Check columns to drop and input data.")
        return None, None, None, None, None, None, None

    print("\nStep 2: Handling outliers using clipping (1st and 99th percentiles for numerical columns)...")
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    for col in numerical_cols:
        if X[col].isnull().all():
            print(f"  Skipping outlier handling for column '{col}' as it is entirely NaN.")
            continue
        if X[col].nunique() <= 1: # Skip if constant after NaNs
            print(f"  Skipping outlier handling for column '{col}' as it has no variance (or is constant).")
            continue

        lower_bound = X[col].quantile(0.01)
        upper_bound = X[col].quantile(0.99)

        if pd.notna(lower_bound) and pd.notna(upper_bound) and lower_bound < upper_bound:
            X.loc[:, col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X.loc[:, col] = np.where(X[col] > upper_bound, upper_bound, X[col])
        elif pd.notna(lower_bound) and pd.notna(upper_bound) and lower_bound >= upper_bound: # Handles case where 0.01 and 0.99 quantiles are same
             print(f"  Skipping outlier handling for column '{col}' as lower bound ({lower_bound}) >= upper bound ({upper_bound}). Column might be mostly constant.")
        else:
            print(f"  Could not calculate valid quantiles for outlier handling in column '{col}'.")

    print("\nStep 3: Identifying and dropping columns with all missing values (post-outlier handling)...")
    cols_all_nan_after_outliers = X.columns[X.isnull().all()].tolist()

    if cols_all_nan_after_outliers:
        print(f"  Dropping columns with all NaN values: {cols_all_nan_after_outliers}")
        X = X.drop(columns=cols_all_nan_after_outliers)
        numerical_cols = [col for col in numerical_cols if col not in cols_all_nan_after_outliers]
    else:
        print("  No columns found with all NaN values at this stage.")

    print(f"Number of features after dropping all-NaN columns: {X.shape[1]}")
    if X.empty:
        print("ERROR: Feature set X is empty after dropping all-NaN columns.")
        return None, None, None, None, None, None, None

    print("\nStep 4: Imputing remaining missing values (median for numerical)...")
    missing_before_imputation = X.isnull().sum()
    missing_before_imputation = missing_before_imputation[missing_before_imputation > 0]
    if not missing_before_imputation.empty:
        print("Missing values per column before imputation (top 15 with missing):")
        print(missing_before_imputation.sort_values(ascending=False).head(15))
    else:
        print("No missing values found in numerical columns before imputation step.")

    if not numerical_cols: # Check if any numerical columns are left
        print("Warning: No numerical columns identified for imputation and scaling. Current features:", X.columns.tolist())
        if X.empty: # Double check, should have been caught earlier
             print("ERROR: Feature set X is empty before imputation.")
             return None, None, None, None, None, None, None
    elif X[numerical_cols].empty:
        print("Warning: Numerical columns subset is empty, cannot impute or scale.")
    else:
        imputer_numerical = SimpleImputer(strategy='median')
        X_imputed_values = imputer_numerical.fit_transform(X[numerical_cols])
        X.loc[:, numerical_cols] = X_imputed_values # Assign back to the original DataFrame slice

    missing_after_imputation = X.isnull().sum().sum()
    print(f"Total missing values in X after imputation: {missing_after_imputation}")
    if missing_after_imputation > 0:
        print("Warning: Missing values still present after imputation. This may occur if non-numerical columns had NaNs or if new NaNs were introduced.")
        print(X.isnull().sum()[X.isnull().sum()>0])


    print("\nStep 5: Scaling numerical features using StandardScaler...")
    if numerical_cols and not X[numerical_cols].empty:
        scaler = StandardScaler()
        X_scaled_values = scaler.fit_transform(X[numerical_cols])
        X.loc[:, numerical_cols] = X_scaled_values # Assign back
    else:
        print("Skipping scaling as there are no numerical columns or numerical subset is empty.")


    final_feature_names = X.columns.tolist()
    print(f"Final number of features after preprocessing: {len(final_feature_names)}")

    print("\nStep 6: Splitting data into training, validation, and test sets (70% train, 15% validation, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp # 0.50 of 0.30 = 0.15
    )
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Validation set shape: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    if y_train is not None and len(y_train)>0:
        print(f"Mortality distribution in training set (before SMOTE):\n{pd.Series(y_train).value_counts(normalize=True)}")
    else:
        print("y_train is empty or None, cannot show distribution.")


    print("\nStep 7: Applying SMOTE to the training data to handle class imbalance...")
    if X_train.empty or y_train.empty:
        print("ERROR: Training data is empty, cannot apply SMOTE.")
        return None, None, None, None, None, None, None

    smote = SMOTE(random_state=42)
    try:
        X_train_smote_values, y_train_smote_values = smote.fit_resample(X_train.values, y_train.values)
        X_train_smote_df = pd.DataFrame(X_train_smote_values, columns=final_feature_names)
        y_train_smote_series = pd.Series(y_train_smote_values, name='mortality')
        print(f"Training set shape after SMOTE: X_train_smote {X_train_smote_df.shape}, y_train_smote {y_train_smote_series.shape}")
        print(f"Mortality distribution in SMOTE-resampled training set:\n{y_train_smote_series.value_counts(normalize=True)}")
    except Exception as e:
        print(f"Error during SMOTE: {e}. Using original training set.")
        X_train_smote_df = X_train.copy()
        y_train_smote_series = y_train.copy()


    X_val_df = X_val.copy()
    X_test_df = X_test.copy()


    print("\nSaving preprocessed datasets...")
    X_train_smote_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'X_train_smote.csv'), index=False)
    y_train_smote_series.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_train_smote.csv'), index=False, header=True)

    X_val_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_val.csv'), index=False, header=['mortality'])

    X_test_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_test.csv'), index=False, header=['mortality'])

    print(f"Preprocessing complete. Datasets saved in '{PREPROCESSED_DATA_PATH}' directory.")

    return X_train_smote_df, y_train_smote_series, X_val_df, y_val, X_test_df, y_test, final_feature_names

if __name__ == "__main__":
    features_df = load_data(FEATURES_FILE)

    if features_df is not None:
        print(f"\nLoaded features_df shape: {features_df.shape}")
        print("\nMissing values in loaded features_df (top 15 with most missing):")
        print(features_df.isnull().sum().sort_values(ascending=False).head(15))

        X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, feature_names = preprocess_data(features_df)

        if X_train_smote is not None and feature_names is not None :
            print("\n--- TODO 3: Data Preprocessing Summary ---")
            print(f"Final number of features used for modeling: {len(feature_names)}")
            print(f"Feature names (first 10 if many): {feature_names[:10]}")
            if len(feature_names) > 10: print("...")