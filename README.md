# AI in EHR - AI Capstone Project: In-Hospital Mortality Prediction

## 1. Project Description

This project focuses on leveraging Electronic Health Records (EHR) from the MIMIC-IV dataset to predict adverse outcomes in ICU patients. The primary goal is to build and evaluate machine learning models for a chosen clinical prediction task.

**Task Chosen**: Task 2 - In-Hospital Mortality Prediction
* Predict in-hospital mortality for ICU patients using vital signs, lab results, diagnoses from prior inpatient visits, and demographic information collected during the early stage (first 6 hours) of ICU admission.

## 2. Dataset

* **Source**: MIMIC-IV (Medical Information Mart for Intensive Care IV)
* **Subset**: A pre-processed subset (approximately 30% of the original population data from Hospital and ICU modules) was provided for this capstone project.
* **TA Pre-processing**:
    * Re-coded `subject_id`.
    * Re-coded admission IDs (`hadm_id`, `stay_id`).
    * Edited original "date" fields with shifts.

## 3. Project Structure

The project is organized into several key scripts and directories:

* `./Data/` - Provided dataset
    * `hosp/`
    * `icu/`
    * `label/`
* `./figures/` - Directory to store generated plots and figures
* `./model_outputs/` - Directory to store trained models and evaluation metrics
* `./preprocessed_data/` - Directory to store data after preprocessing
* `cohort_selection.py` - TODO 1: Script for selecting the study cohort
* `feature_extraction.py` - TODO 2: Script for extracting features and descriptive analysis
* `data_preprocessing.py` - TODO 3: Script for cleaning and preparing data for modeling
* `model_development_logistic_regression.py` - TODO 4: Logistic Regression model
* `model_development_random_forest.py` - TODO 4: Random Forest model
* `model_development_xgboost.py` - TODO 4: XGBoost model
* `final_cohort_task2.csv` - Output of `cohort_selection.py`
* `features_for_mortality_prediction.csv` - Output of `feature_extraction.py`
* `table_one_mortality.csv` - Output of `feature_extraction.py` (likely in `figures/` or main project directory)
* `README.md` - This file
  
## 4. Methodology / Workflow

The project follows these main steps, corresponding to the TODOs in the assignment:

### TODO 1: Cohort Selection (`cohort_selection.py`)

* **Objective**: Select the appropriate study cohort from the provided dataset for Task 2 (In-hospital Mortality Prediction).
* **Criteria**:
    1.  Patients had at least one ICU stay.
    2.  Only the first ICU stay was considered if a patient had multiple ICU stays.
    3.  Patients had at least 6 hours of records in their first ICU stay.
* **Output**:
    * `final_cohort_task2.csv`: A CSV file containing the selected patient cohort.
    * A cohort selection flowchart visualizing the filtering process.

    ![image](https://github.com/user-attachments/assets/81b95c68-44cc-4b42-b638-c9c409f4051a)


### TODO 2: Feature Extraction & Descriptive Analysis (`feature_extraction.py`)

* **Objective**: Extract relevant features and perform descriptive analysis (Table 1).
* **Features Extracted (within the first 6 hours of ICU admission for Task 2, unless specified)**:
    * **Demographics**: Age, Gender (`gender_numeric`).
    * **BMI**: Calculated from height and weight (if available within the first 24 hours of ICU admission).
    * **Vital Signs**: Aggregates (mean, min, max, std, count) for Heart Rate, Respiratory Rate, Mean Arterial Pressure (MAP), Temperature, Systolic Blood Pressure (SBP), Diastolic Blood Pressure (DBP), SpO2.
    * **Laboratory Results**: Aggregates (mean, min, max, std, count, first, last, delta) for BUN, Alkaline Phosphatase, Bilirubin, Creatinine, Glucose, Platelets, Hemoglobin, WBC, Sodium, Potassium, Lactate, Hematocrit, Chloride, Bicarbonate, Anion Gap.
    * **Previous Diagnoses**: Based on the provided `_admissions.csv` subset, this feature indicated whether a patient had distinct prior hospitalizations with diagnoses. Given the dataset characteristics (most patients having only one admission record in the subset), `has_previous_admission_with_dx` was predominantly 'No' (0), and `prev_dx_count_total` was mostly 0.
* **Output**:
    * `features_for_mortality_prediction.csv`: A CSV file with all extracted features merged with the cohort and target variable.
    * `table_one_mortality.csv`: Descriptive statistics (Table 1) comparing features between survivor and non-survivor groups (likely saved in `figures/` by the script).
    * Various feature distribution plots (histograms, boxplots) saved in the `figures/` directory.

### TODO 3: Data Preprocessing (`data_preprocessing.py`)

* **Objective**: Prepare the extracted features for machine learning modeling.
* **Strategies**:
    1.  **Feature Definition**: Dropped identifier columns, potential data leakers (e.g., `los_hours`, `outtime`, `deathtime`, `hospital_expire_flag`), and redundant columns (e.g., original `gender` if `gender_numeric` is used).
    2.  **Outlier Handling**: Clipped numerical features at the 1st and 99th percentiles.
    3.  **Missing Value Imputation**:
        * Dropped columns that were entirely NaN.
        * Imputed remaining missing numerical values using the median.
    4.  **Feature Scaling**: Standardized numerical features using `StandardScaler`.
    5.  **Data Splitting**: Split the data into training (70%), validation (15%), and test (15%) sets, stratified by the 'mortality' target.
    6.  **Handling Class Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the *training set only*.
* **Output**:
    * Preprocessed data splits (`X_train_smote.csv`, `y_train_smote.csv`, `X_test.csv`, `y_test.csv`, etc.) saved in the `preprocessed_data/` directory. Note: The validation set (`X_val`, `y_val`) was generated but might not have been explicitly used by the provided model scripts if they directly use the test set after tuning.

### TODO 4: Model Development & Evaluation

* **Objective**: Build, tune, and evaluate machine learning models to predict in-hospital mortality.
* **Algorithms Used**:
    * Logistic Regression (`model_development_logistic_regression.py`)
    * Random Forest (`model_development_random_forest.py`)
    * XGBoost (`model_development_xgboost.py`)
* **Model Development Strategy**:
    * Hyperparameter tuning using `GridSearchCV` (for Logistic Regression) or `RandomizedSearchCV` (for Random Forest, XGBoost) with 5-fold cross-validation on the SMOTE-resampled training data.
    * Scoring metric for tuning: `roc_auc` or `average_precision` (PR AUC).
* **Evaluation Strategy**:
    * Evaluated the best-tuned model on the held-out test set.
    * Metrics: AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1-Score (for the positive class, i.e., non-survivors/mortality=1), Confusion Matrix.
* **Model Interpretation**:
    * SHAP (SHapley Additive exPlanations) analysis was performed.
    * Coefficients were examined for Logistic Regression.
* **Output (for each model)**:
    * Saved trained model (e.g., `best_logistic_regression_model.joblib`) in `model_outputs/`.
    * Saved evaluation metrics as a CSV file in `model_outputs/`.
    * Saved evaluation plots (ROC curve, PR curve, Confusion Matrix, SHAP plots) in `figures/`.

    #### Logistic Regression Evaluation
    ![image](https://github.com/user-attachments/assets/72fe1732-e3e0-42da-9b08-54465a7e60b3)
    ![image](https://github.com/user-attachments/assets/330867de-3bd1-42b3-b00e-6506647ee1df)
    ![image](https://github.com/user-attachments/assets/04f94d8a-4b91-46e5-8885-ce07d05ac4a9)
    ![image](https://github.com/user-attachments/assets/0212d6d7-20e3-4d7f-9570-29e03856bbea)

    #### Random Forest Evaluation
    ![image](https://github.com/user-attachments/assets/ceca55df-dd99-4699-9dc6-c295a43ce40f)
    ![image](https://github.com/user-attachments/assets/57a526b5-f77a-4c10-9e1f-83d6df7f7767)
    ![image](https://github.com/user-attachments/assets/cd680507-31f4-4bc9-8a5a-0723dbe824a6)
    ![image](https://github.com/user-attachments/assets/6ea8ad22-ba62-4581-abc9-e8204bad2c8e)

    #### XGBoost Evaluation
    ![image](https://github.com/user-attachments/assets/9ba5b24f-8961-4fc5-a4a0-139c5bfb5a82)
    ![image](https://github.com/user-attachments/assets/802506cb-13f1-4798-bad0-1b6288431b7c)
    ![image](https://github.com/user-attachments/assets/feec5846-e144-484f-9b2a-9abaed0ca483)
    ![image](https://github.com/user-attachments/assets/57b12d42-9105-49eb-939c-102253766081)

## 5. How to Run

1.  **Prerequisites**:
    * Python 3.x
    * Install required libraries 
        ```bash
        pip install -r requirements.txt
        ```
2.  **Data Setup**:
    * Place the provided dataset into the `./Data/` directory, maintaining the `hosp/`, `icu/`, and `label/` subfolder structure.
3.  **Run Scripts Sequentially**:
    * **Cohort Selection**:
        ```bash
        python cohort_selection.py
        ```
    * **Feature Extraction**:
        ```bash
        python feature_extraction.py
        ```
    * **Data Preprocessing**:
        ```bash
        python data_preprocessing.py
        ```
    * **Model Development (run each model script)**:
        ```bash
        python model_development_logistic_regression.py
        python model_development_random_forest.py
        python model_development_xgboost.py
        ```

## 6. Results Summary

The models were evaluated on the test set, yielding the following key performance metrics:

| Model                 | Test AUC-ROC | Test AUC-PR | Test Accuracy | Precision (Mortality=1) | Recall (Mortality=1) | F1-Score (Mortality=1) |
| --------------------- | ------------ | ----------- | ------------- | ----------------------- | -------------------- | ---------------------- |
| Logistic Regression   | 0.7745       | 0.4244      | 0.7063        | 0.3574                  | 0.6989               | 0.4729                 |
| Random Forest         | 0.8121       | 0.4838      | 0.8139        | 0.5093                  | 0.3453               | 0.4115                 |
| XGBoost               | 0.8305       | 0.5507      | 0.8302        | 0.5929                  | 0.3158               | 0.4121                 |

* **Best Overall Performance (AUCs)**: XGBoost demonstrated the highest AUC-ROC (0.8305) and AUC-PR (0.5507), indicating better overall discrimination and performance on the imbalanced dataset.
* **Precision vs. Recall**:
    * Logistic Regression achieved the highest recall for the mortality class (0.6989), meaning it identified a larger proportion of actual mortality cases, but at the cost of lower precision (0.3574).
    * XGBoost achieved the highest precision (0.5929), meaning that when it predicted mortality, it was more likely to be correct, but it had a lower recall (0.3158) compared to Logistic Regression.
* **Accuracy**: XGBoost also had the highest accuracy (0.8302).

* **Feature Importance (XGBoost - Best Model)**:
    SHAP analysis for the XGBoost model highlighted several key predictors for in-hospital mortality:
    * **`age`** was the most influential feature, underscoring its strong association with patient outcomes.
    * Counts of previous diagnoses by chapter, such as **`prev_dx_respiratory_count`**, **`prev_dx_other_icd9_count`**, and **`prev_dx_circulatory_count`**, were highly important. This suggests that the burden and type of past comorbidities significantly impact mortality risk, even when derived from a limited history in the provided admissions data.
    * Vital signs from the first 6 hours of ICU stay, including **`temperature_count`** (frequency of measurement), **`sbp_min`** (minimum systolic blood pressure), and various aggregates of **`respiratory_rate`**, were also significant contributors.
    * Other notable features included `gender_numeric`, `spo2_min` (minimum oxygen saturation), and `lactate_count`.
    The SHAP analysis indicates that a combination of demographic factors (age), historical health status (previous diagnoses counts), and acute physiological derangements in early ICU stay (vital signs) are critical for predicting mortality.

## 7. Conclusion

This project successfully developed and evaluated machine learning models for predicting in-hospital mortality using early ICU data from a MIMIC-IV subset. Among the models tested, **XGBoost** provided the best balance of discriminative ability (AUC-ROC, AUC-PR) and precision for predicting mortality. Logistic Regression, while having lower overall AUCs, showed strength in recall for the minority (mortality) class.

The performance, while reasonable, is likely constrained by the 30% data subset used. The feature `has_previous_admission_with_dx` had limited variability due to most patients in the provided `_admissions.csv` subset having only one recorded admission, which limited its utility in capturing extensive prior history for this specific dataset.

**Limitations**:
* The dataset used was a subset, potentially limiting the generalizability and maximum achievable performance.
* Feature engineering was based on aggregates within the first 6 hours; more complex temporal patterns were not explored.
* The `_admissions.csv` subset limited the ability to extract extensive features related to distinct prior hospitalizations for most patients.

**Future Work**:
* Train models on the full MIMIC-IV dataset if accessible.
* Explore advanced feature engineering techniques, including more granular temporal features or interaction terms.
* Investigate more complex models, such as Recurrent Neural Networks (RNNs/LSTMs) or Transformers, which can better capture temporal sequences in EHR data.
* Conduct a more in-depth error analysis to understand specific patient subgroups where models perform poorly.
* If a more complete admissions history were available, re-evaluate the impact of previous hospitalizations.

## 8. References

* **About the data (MIMIC-IV)**:
    1.  MIMIC official site: [https://mimic.mit.edu/](https://mimic.mit.edu/)
    2.  Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. [https://doi.org/10.13026/6mm1-ek60](https://doi.org/10.13026/6mm1-ek60).
* **About the task 2 (in-hospital mortality) - *Selected from assignment***:
    1.  Gao, J., Lu, Y., Ashrafi, N., Domingo, I., Alaei, K., & Pishgar, M. (2024). Prediction of sepsis mortality in ICU patients using machine learning methods. *BMC Medical Informatics and Decision Making, 24*(1), 228.
    2.  Iwase, S., Nakada, T.-A., Shimada, T., Oami, T., Shimazui, T., Takahashi, N., Yamabe, J., Yamao, Y., & Kawakami, E. (2022). Prediction algorithm for ICU mortality and length of stay using machine learning. *Scientific Reports, 12*(1), 12912.
    3.  Hou, N., Li, M., He, L., Xie, B., Wang, L., Zhang, R., Yu, Y., Sun, X., Pan, Z., & Wang, K. (2020). Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost. *Journal of Translational Medicine, 18*(1), 462.

