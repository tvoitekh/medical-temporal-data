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
        print(f"ERROR: File not found at {file_path}. Please ensure TODO 2 was run successfully.")
        return None
    return pd.read_csv(file_path)

def preprocess_data(df):
    if df is None:
        return None, None, None, None, None, None, None

    print("\nStarting data preprocessing...")

    print("Step 1: Defining target and features...")
    if 'mortality' not in df.columns:
        print("ERROR: Target column 'mortality' not found.")
        return None, None, None, None, None, None, None
    
    y = df['mortality']
    
    cols_to_drop = [
        'subject_id', 'hadm_id', 'stay_id', 'gender', 
        'intime', 'outtime', 'deathtime', 
        'los_icu',
        'los',
        'los_hours',
        'hospital_expire_flag',
        'first_careunit', 'last_careunit'
    ] 
    
    X = df.drop(columns=['mortality'] + [col for col in cols_to_drop if col in df.columns])
    print(f"Number of features after initial drop: {X.shape[1]}")
    
    print("\nStep 2: Handling outliers using clipping (1st and 99th percentiles)...")
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    for col in numerical_cols:
        if X[col].isnull().all():
            print(f"  Skipping outlier handling for column '{col}' as it is entirely NaN.")
            continue
        lower_bound = X[col].quantile(0.01)
        upper_bound = X[col].quantile(0.99)
        if pd.notna(lower_bound) and pd.notna(upper_bound):
            X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
        else:
            print(f"  Could not calculate quantiles for outlier handling in column '{col}' (possibly too many NaNs or constant).")

    print("\nStep 3: Identifying and dropping columns with all missing values...")
    cols_all_nan = []
    for col in X.columns:
        if X[col].isnull().all():
            cols_all_nan.append(col)
            
    if cols_all_nan:
        print(f"  Dropping columns with all NaN values: {cols_all_nan}")
        X = X.drop(columns=cols_all_nan)
        numerical_cols = [col for col in numerical_cols if col not in cols_all_nan]
    else:
        print("  No columns found with all NaN values.")
    
    print(f"Number of features after dropping all-NaN columns: {X.shape[1]}")

    print("\nStep 4: Imputing remaining missing values (median for numerical)...")
    missing_before = X.isnull().sum()
    missing_before = missing_before[missing_before > 0]
    if not missing_before.empty:
        print("Missing values before imputation (after all-NaN drop):\n", missing_before.sort_values(ascending=False).head(15))
    else:
        print("No missing values found before imputation step (after all-NaN drop).")

    if not numerical_cols:
        print("ERROR: No numerical columns left for imputation.")
        return None, None, None, None, None, None, None
        
    imputer_numerical = SimpleImputer(strategy='median')
    X_imputed_numerical = imputer_numerical.fit_transform(X[numerical_cols])
    X = pd.DataFrame(X_imputed_numerical, columns=numerical_cols, index=X.index)

    missing_after = X.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")
    if missing_after > 0:
        print("Warning: Missing values still present after imputation.")

    print("\nStep 5: Scaling numerical features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numerical_cols])
    X = pd.DataFrame(X_scaled, columns=numerical_cols, index=X.index)

    final_feature_names = X.columns.tolist()
    print(f"Final number of features after preprocessing: {len(final_feature_names)}")

    print("\nStep 6: Splitting data into training, validation, and test sets (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Validation set shape: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"Mortality distribution in training set (before SMOTE):\n{pd.Series(y_train).value_counts(normalize=True)}")

    print("\nStep 7: Applying SMOTE to the training data to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    X_train_smote_values, y_train_smote_values = smote.fit_resample(X_train_values, y_train_values)
    
    X_train_smote_df = pd.DataFrame(X_train_smote_values, columns=final_feature_names)
    y_train_smote_series = pd.Series(y_train_smote_values, name='mortality')

    print(f"Training set shape after SMOTE: X_train_smote {X_train_smote_df.shape}, y_train_smote {y_train_smote_series.shape}")
    print(f"Mortality distribution in SMOTE-resampled training set:\n{y_train_smote_series.value_counts(normalize=True)}")

    X_val_df = pd.DataFrame(X_val, columns=final_feature_names) if not isinstance(X_val, pd.DataFrame) else X_val.copy()
    X_test_df = pd.DataFrame(X_test, columns=final_feature_names) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
    if X_val_df.columns.tolist() != final_feature_names: X_val_df.columns = final_feature_names
    if X_test_df.columns.tolist() != final_feature_names: X_test_df.columns = final_feature_names

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
        print("\nMissing values in loaded features_df (top 15):")
        print(features_df.isnull().sum().sort_values(ascending=False).head(15))
        
        if 'los' in features_df.columns:
            print("\n'los' column statistics (potential data leaker for current ICU mortality):")
            print(features_df['los'].describe())
        if 'los_hours' in features_df.columns:
            print("\n'los_hours' column statistics (potential data leaker for current ICU mortality):")
            print(features_df['los_hours'].describe())

        X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, feature_names = preprocess_data(features_df)

        if X_train_smote is not None:
            print("\n--- TODO 3: Data Preprocessing Summary ---")
            print(f"Final number of features used for modeling: {len(feature_names)}")
            print(f"Feature names: {feature_names}")