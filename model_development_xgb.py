import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix
)
import shap
import joblib

PREPROCESSED_DATA_PATH = './preprocessed_data/'
MODEL_OUTPUT_PATH = './model_outputs/'
if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)
FIGURES_PATH = './figures/'
if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

def load_preprocessed_data():
    print("Loading preprocessed data...")
    try:
        X_train = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, 'X_train_smote.csv'))
        y_train = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_train_smote.csv'))['mortality']
        X_test = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_test.csv'))['mortality']
        print("Data loaded successfully.")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure TODO 3 was run successfully and files are in '{PREPROCESSED_DATA_PATH}'.")
        return None, None, None, None

def plot_roc_curve(y_true, y_pred_proba, model_name_suffix):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name_suffix}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_PATH, f'roc_curve_{model_name_suffix.replace(" ", "_").lower()}.png'))
    plt.close()

def plot_pr_curve(y_true, y_pred_proba, model_name_suffix):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title(f'Precision-Recall Curve - {model_name_suffix}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(FIGURES_PATH, f'pr_curve_{model_name_suffix.replace(" ", "_").lower()}.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name_suffix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Survivor (0)', 'Non-Survivor (1)'],
                yticklabels=['Survivor (0)', 'Non-Survivor (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name_suffix}')
    plt.savefig(os.path.join(FIGURES_PATH, f'confusion_matrix_{model_name_suffix.replace(" ", "_").lower()}.png'))
    plt.close()

def evaluate_model(model, X_test_data, y_test_data, model_name_suffix="Model", threshold=0.5):
    print(f"\nEvaluating {model_name_suffix} on the test set...")
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_data, y_pred_proba)
    auc_pr_val = auc(recall_vals, precision_vals)

    metrics = {
        "Model": model_name_suffix,
        "AUC-ROC": roc_auc_score(y_test_data, y_pred_proba),
        "AUC-PR": auc_pr_val,
        "Accuracy": accuracy_score(y_test_data, y_pred_binary),
        "Precision (Non-Survivor)": precision_score(y_test_data, y_pred_binary, pos_label=1, zero_division=0),
        "Recall (Non-Survivor)": recall_score(y_test_data, y_pred_binary, pos_label=1, zero_division=0),
        "F1-Score (Non-Survivor)": f1_score(y_test_data, y_pred_binary, pos_label=1, zero_division=0)
    }
    print(f"Metrics for {model_name_suffix}:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"  {metric}: {value:.4f}")

    plot_roc_curve(y_test_data, y_pred_proba, model_name_suffix)
    plot_pr_curve(y_test_data, y_pred_proba, model_name_suffix)
    plot_confusion_matrix(y_test_data, y_pred_binary, model_name_suffix)
    
    return metrics

def perform_shap_analysis(model, X_data, model_name_suffix="Model"):
    print(f"\nPerforming SHAP analysis for {model_name_suffix}...")
    try:
        if isinstance(model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data)
            shap_values_positive_class = shap_values
        else:
            print(f"SHAP analysis for {type(model)} might require a different explainer or setup.")
            return

        plt.figure()
        shap.summary_plot(shap_values_positive_class, X_data, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance - {model_name_suffix}")
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_bar_{model_name_suffix.replace(" ", "_").lower()}.png'), bbox_inches='tight')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values_positive_class, X_data, show=False, max_display=20)
        plt.title(f"SHAP Summary Plot - {model_name_suffix}")
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_dot_{model_name_suffix.replace(" ", "_").lower()}.png'), bbox_inches='tight')
        plt.close()
        
        print(f"SHAP analysis plots for {model_name_suffix} saved.")
    except Exception as e:
        print(f"Could not perform SHAP analysis for {model_name_suffix}: {e}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_preprocessed_data()

    if X_train is None:
        print("Exiting due to data loading failure.")
        exit()

    model_name_xgb = "XGBoost"
    print(f"\n--- Training {model_name_xgb} Model ---")
    
    xgb_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_random_search = RandomizedSearchCV(
        xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False, # This is deprecated, consider objective below
            eval_metric='logloss',
            tree_method='hist',
            device='cuda:1'
        ),
        param_distributions=xgb_param_dist,
        n_iter=25,
        cv=5,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    print(f"Starting RandomizedSearchCV for {model_name_xgb}...")
    xgb_random_search.fit(X_train, y_train)

    best_xgb_model = xgb_random_search.best_estimator_
    print(f"\nBest {model_name_xgb} parameters: {xgb_random_search.best_params_}")
    joblib.dump(best_xgb_model, os.path.join(MODEL_OUTPUT_PATH, 'best_xgboost_model.joblib'))
    print(f"Best {model_name_xgb} model saved to '{os.path.join(MODEL_OUTPUT_PATH, 'best_xgboost_model.joblib')}'")

    xgb_metrics = evaluate_model(best_xgb_model, X_test, y_test, model_name_xgb)

    xgb_metrics_df = pd.DataFrame([xgb_metrics])
    xgb_metrics_df.to_csv(os.path.join(MODEL_OUTPUT_PATH, 'xgboost_evaluation_metrics.csv'), index=False)
    print(f"{model_name_xgb} evaluation metrics saved to '{os.path.join(MODEL_OUTPUT_PATH, 'xgboost_evaluation_metrics.csv')}'")
    
    perform_shap_analysis(best_xgb_model, X_test, model_name_xgb)
    
    print(f"\n--- {model_name_xgb} Model Script Finished ---")