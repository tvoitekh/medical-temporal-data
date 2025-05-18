import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
import shap # Optional: for feature importance, ensure it's installed

# Paths based on our project structure
PREPROCESSED_DATA_PATH = './preprocessed_data/'
MODEL_OUTPUT_PATH = './model_outputs/'
FIGURES_PATH = './figures/'

# Specific file names from our preprocessing script
TRAIN_X_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'X_train_smote.csv')
TRAIN_Y_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'y_train_smote.csv')
TEST_X_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'X_test.csv')
TEST_Y_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'y_test.csv')

LOGISTIC_MODEL_FILENAME = 'logistic_regression_model.pkl'
LOGISTIC_MODEL_PATH = os.path.join(MODEL_OUTPUT_PATH, LOGISTIC_MODEL_FILENAME)

def ensure_dir_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def load_data():
    print("Loading preprocessed data...")
    try:
        X_train = pd.read_csv(TRAIN_X_PATH)
        # y_train might be a DataFrame with one column or a Series if saved with header=True
        y_train_df = pd.read_csv(TRAIN_Y_PATH)
        y_train = y_train_df.squeeze() # Handles both Series and single-column DataFrame

        X_test = pd.read_csv(TEST_X_PATH)
        y_test_df = pd.read_csv(TEST_Y_PATH)
        y_test = y_test_df.squeeze()

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        if not y_train.empty:
            print(f"y_train distribution:\n{y_train.value_counts(normalize=True).to_string()}")
        else:
            print("y_train is empty after loading.")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure preprocessing was run successfully.")
        return None, None, None, None
    except Exception as e_gen:
        print(f"An unexpected error occurred during data loading: {e_gen}")
        traceback.print_exc()
        return None, None, None, None

def plot_evaluation_curves(estimator, X_data, y_data, model_name_suffix):
    ensure_dir_exists(FIGURES_PATH)
    model_name_safe = model_name_suffix.replace(" ", "_").lower()

    # ROC Curve
    try:
        RocCurveDisplay.from_estimator(estimator, X_data, y_data, name=model_name_suffix)
        plt.title(f'ROC Curve - {model_name_suffix}')
        plt.savefig(os.path.join(FIGURES_PATH, f'roc_curve_{model_name_safe}.png'))
        plt.close()
        print(f"ROC curve for {model_name_suffix} saved.")
    except Exception as e:
        print(f"Could not plot ROC curve for {model_name_suffix}: {e}")

    # Precision-Recall Curve
    try:
        PrecisionRecallDisplay.from_estimator(estimator, X_data, y_data, name=model_name_suffix)
        plt.title(f'Precision-Recall Curve - {model_name_suffix}')
        plt.savefig(os.path.join(FIGURES_PATH, f'pr_curve_{model_name_safe}.png'))
        plt.close()
        print(f"Precision-Recall curve for {model_name_suffix} saved.")
    except Exception as e:
        print(f"Could not plot PR curve for {model_name_suffix}: {e}")


def plot_custom_confusion_matrix(y_true, y_pred, model_name_suffix):
    ensure_dir_exists(FIGURES_PATH)
    model_name_safe = model_name_suffix.replace(" ", "_").lower()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Survivor (0)', 'Non-Survivor (1)'],
                yticklabels=['Survivor (0)', 'Non-Survivor (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name_suffix}')
    plt.savefig(os.path.join(FIGURES_PATH, f'confusion_matrix_{model_name_safe}.png'))
    plt.close()
    print(f"Confusion matrix for {model_name_suffix} saved.")
    return cm

def perform_shap_analysis(model, X_data_df, model_name_suffix="Model"):
    ensure_dir_exists(FIGURES_PATH)
    model_name_safe = model_name_suffix.replace(" ", "_").lower()
    print(f"\nPerforming SHAP analysis for {model_name_suffix}...")
    try:
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_data_df, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_data_df)
        else:
            print(f"SHAP analysis for {type(model)} might require a different explainer or setup. Skipping.")
            return

        plt.figure()
        shap.summary_plot(shap_values, X_data_df, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance (Bar) - {model_name_suffix}")
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_bar_{model_name_safe}.png'), bbox_inches='tight')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_data_df, show=False, max_display=20) # Dot plot
        plt.title(f"SHAP Summary Plot (Dots) - {model_name_suffix}")
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_dot_{model_name_safe}.png'), bbox_inches='tight')
        plt.close()

        print(f"SHAP analysis plots for {model_name_suffix} saved.")

        if hasattr(model, 'coef_'):
            coefficients = pd.DataFrame(model.coef_[0], X_data_df.columns, columns=['Coefficient'])
            print("\nModel Coefficients (Top 10 by absolute value):")
            print(coefficients.reindex(coefficients.Coefficient.abs().sort_values(ascending=False).index).head(10))
        else:
            print("Model does not have 'coef_' attribute.")

    except Exception as e:
        print(f"Could not perform SHAP analysis or display coefficients for {model_name_suffix}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Logistic Regression Model Training and Evaluation...")

    ensure_dir_exists(MODEL_OUTPUT_PATH)
    ensure_dir_exists(FIGURES_PATH)

    X_train, y_train, X_test, y_test = load_data()

    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Exiting due to data loading failure.")
        exit()

    log_reg_model = LogisticRegression(
        solver='liblinear',
        random_state=42,
        class_weight='balanced',
        max_iter=2000 # Increased max_iter
    )

    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    print("\nPerforming GridSearchCV for Logistic Regression...")
    grid_search_lr = GridSearchCV(
        estimator=log_reg_model,
        param_grid=param_grid_lr,
        cv=5,
        scoring='average_precision', # PR AUC, as per your example
        verbose=1,
        n_jobs=-1
    )
    try:
        grid_search_lr.fit(X_train, y_train)
        print(f"Best parameters found: {grid_search_lr.best_params_}")
        print(f"Best average_precision score on CV: {grid_search_lr.best_score_:.4f}")
        best_log_reg_model = grid_search_lr.best_estimator_
    except Exception as e:
        print(f"Error during GridSearchCV: {e}")
        print("Falling back to default Logistic Regression model.")
        traceback.print_exc()
        best_log_reg_model = LogisticRegression(
            solver='liblinear',
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            C=1.0, # Default C
            penalty='l2' # Default penalty
        ).fit(X_train, y_train)


    print("\nEvaluating the best model on the test set...")
    y_pred_proba_lr = best_log_reg_model.predict_proba(X_test)[:, 1]
    y_pred_lr = best_log_reg_model.predict(X_test)

    roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
    pr_auc_lr = auc(recall_lr, precision_lr)

    print(f"Test Set ROC AUC: {roc_auc_lr:.4f}")
    print(f"Test Set PR AUC (Average Precision): {pr_auc_lr:.4f}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_lr, zero_division=0))

    print("\nTest Set Confusion Matrix:")
    cm_lr = plot_custom_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
    tn, fp, fn, tp = cm_lr.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    plot_evaluation_curves(best_log_reg_model, X_test, y_test, "Logistic Regression")

    print(f"\nSaving the best Logistic Regression model to {LOGISTIC_MODEL_PATH}...")
    try:
        with open(LOGISTIC_MODEL_PATH, 'wb') as f:
            pickle.dump(best_log_reg_model, f)
        print("Model saved.")
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()

    perform_shap_analysis(best_log_reg_model, X_test, "Logistic Regression")

    print("\nLogistic Regression script finished.")
