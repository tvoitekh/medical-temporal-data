import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

BASE_PATH = './Data/'
ICU_PATH = BASE_PATH + 'icu/'
HOSP_PATH = BASE_PATH + 'hosp/'
LABEL_PATH = BASE_PATH + 'label/'
FIGURES_PATH = './figures/'
if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

try:
    final_cohort = pd.read_csv('final_cohort.csv')
    print(f"Loaded cohort with {final_cohort.shape[0]} patients from final_cohort.csv")
except FileNotFoundError:
    print("final_cohort.csv not found. Please run TODO 1 script first.")
    exit()

def load_chartevents():
    print("Loading chart events data...")
    return pd.read_csv(ICU_PATH + '_chartevents.csv')

def load_labevents():
    print("Loading lab events data...")
    return pd.read_csv(HOSP_PATH + '_labevents.csv')

def load_diagnoses():
    print("Loading diagnoses data...")
    return pd.read_csv(HOSP_PATH + '_diagnoses_icd.csv')

def calculate_bmi(height_cm, weight_kg):
    if pd.isna(height_cm) or pd.isna(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return np.nan
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)

def extract_demographics():
    print("Extracting demographics...")
    
    patients = pd.read_csv(HOSP_PATH + '_patients.csv')
    cohort_patients = patients[patients['subject_id'].isin(final_cohort['subject_id'])].copy()
    
    gender_map = {'M': 1, 'F': 0}
    cohort_patients['gender_numeric'] = cohort_patients['gender'].map(gender_map)
    
    admissions = pd.read_csv(HOSP_PATH + '_admissions.csv')
    cohort_admissions = admissions[admissions['hadm_id'].isin(final_cohort['hadm_id'])].copy()

    demographics = pd.merge(
        cohort_patients[['subject_id', 'gender', 'gender_numeric', 'anchor_year', 'anchor_age']], 
        cohort_admissions[['subject_id', 'hadm_id', 'admittime']], 
        on='subject_id',
        how='inner'
    )
    demographics = demographics[demographics['hadm_id'].isin(final_cohort['hadm_id'])]

    demographics['admittime_year'] = pd.to_datetime(demographics['admittime']).dt.year
    demographics['age'] = demographics['anchor_age'] + (demographics['admittime_year'] - demographics['anchor_year'])
    
    demographics.loc[demographics['age'] > 90, 'age'] = 90

    demographics = demographics[['subject_id', 'hadm_id', 'gender', 'gender_numeric', 'age']].drop_duplicates()
    
    print(f"Extracted demographics for {demographics['subject_id'].nunique()} unique subjects, {demographics['hadm_id'].nunique()} unique hospital admissions.")
    return demographics

def extract_bmi():
    print("Extracting BMI...")
    
    chartevents = load_chartevents()
    
    cohort_stays = final_cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    cohort_stays['intime'] = pd.to_datetime(cohort_stays['intime'])
    cohort_stays['cutoff_time'] = cohort_stays['intime'] + pd.Timedelta(hours=24) 
    
    chartevents_cohort = chartevents[chartevents['stay_id'].isin(cohort_stays['stay_id'])].copy()
    chartevents_cohort['charttime'] = pd.to_datetime(chartevents_cohort['charttime'])
    
    chartevents_cohort = pd.merge(
        chartevents_cohort,
        cohort_stays[['stay_id', 'intime', 'cutoff_time']],
        on='stay_id',
        how='left'
    )
    
    chartevents_bmi_window = chartevents_cohort[(chartevents_cohort['charttime'] >= chartevents_cohort['intime']) & 
                                                (chartevents_cohort['charttime'] <= chartevents_cohort['cutoff_time'])].copy()
    
    height_ids = [226707, 226730]
    weight_ids = [226512, 224639, 226531]

    heights = chartevents_bmi_window[chartevents_bmi_window['itemid'].isin(height_ids)].copy()
    weights = chartevents_bmi_window[chartevents_bmi_window['itemid'].isin(weight_ids)].copy()
    
    heights = heights.sort_values(['stay_id', 'charttime']).groupby('stay_id').first().reset_index()
    weights = weights.sort_values(['stay_id', 'charttime']).groupby('stay_id').first().reset_index()
    
    bmi_data = pd.merge(
        heights[['subject_id', 'hadm_id', 'stay_id', 'valuenum']],
        weights[['stay_id', 'valuenum']],
        on='stay_id',
        how='inner',
        suffixes=('_height', '_weight')
    )
    
    bmi_data['bmi'] = bmi_data.apply(
        lambda x: calculate_bmi(x['valuenum_height'], x['valuenum_weight']), axis=1)
    
    bmi_data = bmi_data[(bmi_data['bmi'] >= 10) & (bmi_data['bmi'] <= 70)] 
    
    bmi_data = bmi_data[['subject_id', 'hadm_id', 'stay_id', 'bmi']]
    
    print(f"Extracted BMI for {bmi_data['stay_id'].nunique()} ICU stays.")
    return bmi_data

def extract_vitals():
    print("Extracting vital signs...")
    
    chartevents = load_chartevents()
    
    chartevents_cohort = chartevents[chartevents['stay_id'].isin(final_cohort['stay_id'])].copy()
    chartevents_cohort['charttime'] = pd.to_datetime(chartevents_cohort['charttime'])
    
    cohort_times = final_cohort[['stay_id', 'intime']].copy()
    cohort_times['intime'] = pd.to_datetime(cohort_times['intime'])
    cohort_times['cutoff_time'] = cohort_times['intime'] + pd.Timedelta(hours=6)
    
    chartevents_cohort = pd.merge(
        chartevents_cohort,
        cohort_times[['stay_id', 'intime', 'cutoff_time']],
        on='stay_id',
        how='left'
    )
    
    chartevents_first_6h = chartevents_cohort[(chartevents_cohort['charttime'] >= chartevents_cohort['intime']) & 
                                             (chartevents_cohort['charttime'] <= chartevents_cohort['cutoff_time'])].copy()
    
    vital_ids = {
        220045: 'heart_rate',
        220210: 'respiratory_rate',
        220052: 'map',
        223761: 'temperature_c',
        223762: 'temperature_f',
        220179: 'sbp',
        220180: 'dbp',
        220277: 'spo2'
    }
    
    vitals = chartevents_first_6h[chartevents_first_6h['itemid'].isin(vital_ids.keys())].copy()
    vitals['vital_name'] = vitals['itemid'].map(vital_ids)

    if 'temperature_f' in vitals['vital_name'].unique() and 'temperature_c' not in vitals['vital_name'].unique():
        vitals.loc[vitals['vital_name'] == 'temperature_f', 'valuenum'] = (vitals['valuenum'] - 32) * 5/9
        vitals.loc[vitals['vital_name'] == 'temperature_f', 'vital_name'] = 'temperature'
    elif 'temperature_f' in vitals['vital_name'].unique() and 'temperature_c' in vitals['vital_name'].unique():
        temp_f_present = vitals['vital_name'] == 'temperature_f'
        vitals.loc[temp_f_present, 'valuenum'] = (vitals.loc[temp_f_present, 'valuenum'] - 32) * 5/9
        vitals.loc[temp_f_present, 'vital_name'] = 'temperature'
        vitals.loc[vitals['vital_name'] == 'temperature_c', 'vital_name'] = 'temperature'
    elif 'temperature_c' in vitals['vital_name'].unique():
         vitals.loc[vitals['vital_name'] == 'temperature_c', 'vital_name'] = 'temperature'

    vital_features_list = []
    for vital_name_key, vital_name_val in vital_ids.items():
        current_vital_name = 'temperature' if 'temperature' in vital_name_val else vital_name_val
        
        if vital_name_val == 'temperature_f' and 'temperature_c' in vital_ids.values():
            continue

        vital_data = vitals[vitals['vital_name'] == current_vital_name]
        if not vital_data.empty:
            agg_funcs = ['mean', 'min', 'max', 'std', 'count']

            summary = vital_data.groupby(['subject_id', 'hadm_id', 'stay_id'])['valuenum'].agg(agg_funcs).reset_index()
            summary.columns = ['subject_id', 'hadm_id', 'stay_id'] + [f'{current_vital_name}_{func}' for func in agg_funcs]
            vital_features_list.append(summary)

    if not vital_features_list:
        print("No vital signs data found for the cohort in the first 6 hours.")
        return final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    vital_features_final = final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
    for df_vital in vital_features_list:
        vital_features_final = pd.merge(vital_features_final, df_vital, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    print(f"Extracted vital signs aggregates for {vital_features_final['stay_id'].nunique()} ICU stays.")
    return vital_features_final

def extract_labs():
    print("Extracting laboratory results...")
    
    labevents = load_labevents()
    
    labevents_cohort = labevents[labevents['hadm_id'].isin(final_cohort['hadm_id'])].copy()
    labevents_cohort['charttime'] = pd.to_datetime(labevents_cohort['charttime'])
    
    lab_ids = {
        51006: 'bun',
        50863: 'alkaline_phosphatase',
        50885: 'bilirubin_total',
        50912: 'creatinine',
        50931: 'glucose_lab',
        51265: 'platelets',
        51222: 'hemoglobin',
        51301: 'wbc',
        50824: 'sodium',
        50971: 'potassium',
        50809: 'glucose_bg',
        50818: 'lactate',
        51221: 'hematocrit',
        50902: 'chloride',
        50882: 'bicarbonate',
        50868: 'anion_gap'
    }
    
    labs = labevents_cohort[labevents_cohort['itemid'].isin(lab_ids.keys())].copy()
    labs['lab_name'] = labs['itemid'].map(lab_ids)
    
    icu_stay_times = final_cohort[['hadm_id', 'stay_id', 'intime']].drop_duplicates(subset=['hadm_id', 'stay_id'])
    icu_stay_times['intime'] = pd.to_datetime(icu_stay_times['intime'])
    icu_stay_times['cutoff_time'] = icu_stay_times['intime'] + pd.Timedelta(hours=6)

    labs = pd.merge(labs, icu_stay_times, on='hadm_id', how='left')
    labs = labs.dropna(subset=['stay_id']) 
    labs_first_6h = labs[(labs['charttime'] >= labs['intime']) & 
                         (labs['charttime'] <= labs['cutoff_time'])].copy()

    lab_features_list = []
    for lab_id_key, lab_name_val in lab_ids.items():
        lab_data = labs_first_6h[labs_first_6h['lab_name'] == lab_name_val]
        if not lab_data.empty:
            agg_funcs = ['mean', 'min', 'max', 'std', 'count', 'first', 'last']
            summary = lab_data.groupby(['subject_id', 'hadm_id', 'stay_id'])['valuenum'].agg(agg_funcs).reset_index()
            
            new_cols = ['subject_id', 'hadm_id', 'stay_id']
            for func in agg_funcs:
                new_cols.append(f'{lab_name_val}_{func}')
            summary.columns = new_cols
            
            if f'{lab_name_val}_first' in summary.columns and f'{lab_name_val}_last' in summary.columns:
                summary[f'{lab_name_val}_delta'] = summary[f'{lab_name_val}_last'] - summary[f'{lab_name_val}_first']
            
            lab_features_list.append(summary)

    if not lab_features_list:
        print("No lab results data found for the cohort in the first 6 hours.")
        return final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    lab_features_final = final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
    for df_lab in lab_features_list:
        lab_features_final = pd.merge(lab_features_final, df_lab, on=['subject_id', 'hadm_id', 'stay_id'], how='left')

    print(f"Extracted lab results aggregates for {lab_features_final['stay_id'].nunique()} ICU stays.")
    return lab_features_final

def extract_previous_diagnoses():
    print("Extracting previous diagnoses...")
    
    diagnoses_icd = load_diagnoses()
    admissions = pd.read_csv(HOSP_PATH + '_admissions.csv')
    
    current_admissions_info = final_cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    current_admissions_info['intime'] = pd.to_datetime(current_admissions_info['intime'])
    
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

    all_previous_diagnoses_dfs = []

    for idx, current_stay in current_admissions_info.iterrows():
        subject_id = current_stay['subject_id']
        current_hadm_id = current_stay['hadm_id']
        current_intime = current_stay['intime']
        
        subject_all_admissions = admissions[admissions['subject_id'] == subject_id].sort_values(by='admittime')
        previous_hospitalizations = subject_all_admissions[subject_all_admissions['admittime'] < current_intime]
        
        if not previous_hospitalizations.empty:
            prev_hadm_ids = previous_hospitalizations['hadm_id'].unique()
            dx_for_prev_hadms = diagnoses_icd[diagnoses_icd['hadm_id'].isin(prev_hadm_ids)].copy()
            
            if not dx_for_prev_hadms.empty:
                dx_for_prev_hadms['stay_id'] = current_stay['stay_id']
                all_previous_diagnoses_dfs.append(dx_for_prev_hadms)

    if not all_previous_diagnoses_dfs:
        print("No previous hospitalizations with diagnoses found for any patient in the cohort.")
        prev_dx_features = final_cohort[['subject_id','hadm_id','stay_id']].drop_duplicates().copy()
        prev_dx_features['has_previous_admission_with_dx'] = 0
        prev_dx_features['prev_dx_count_total'] = 0
        return prev_dx_features

    all_prev_dx_df = pd.concat(all_previous_diagnoses_dfs)

    def map_icd_to_chapter(icd_code, icd_version):
        if pd.isna(icd_code): return 'unknown'
        code = str(icd_code).upper()
        if icd_version == 9:
            if 'V' in code or 'E' in code: return 'other_icd9'
            try:
                num_part = int(re.match(r"([0-9]+)", code).group(1))
                if 390 <= num_part <= 459: return 'circulatory'
                if 460 <= num_part <= 519: return 'respiratory'
                if 240 <= num_part <= 279: return 'endocrine'
            except: return 'other_icd9'
        elif icd_version == 10:
            if code.startswith('I'): return 'circulatory'
            if code.startswith('J'): return 'respiratory'
            if code.startswith('E'): return 'endocrine'
        return 'other_icd'

    all_prev_dx_df['icd_chapter'] = all_prev_dx_df.apply(lambda row: map_icd_to_chapter(row['icd_code'], row['icd_version']), axis=1)

    prev_dx_total_counts = all_prev_dx_df.groupby('stay_id').size().reset_index(name='prev_dx_count_total')

    prev_dx_chapter_counts = all_prev_dx_df.groupby(['stay_id', 'icd_chapter']).size().unstack(fill_value=0)
    prev_dx_chapter_counts.columns = [f'prev_dx_{col}_count' for col in prev_dx_chapter_counts.columns]
    prev_dx_chapter_counts = prev_dx_chapter_counts.reset_index()

    prev_dx_features = pd.merge(final_cohort[['subject_id','hadm_id','stay_id']].drop_duplicates(), prev_dx_total_counts, on='stay_id', how='left')
    prev_dx_features = pd.merge(prev_dx_features, prev_dx_chapter_counts, on='stay_id', how='left')
    
    prev_dx_features['has_previous_admission_with_dx'] = np.where(prev_dx_features['prev_dx_count_total'] > 0, 1, 0)
    prev_dx_features = prev_dx_features.fillna(0)

    print(f"Extracted previous diagnoses features for {prev_dx_features['stay_id'].nunique()} ICU stays.")
    return prev_dx_features


def create_table_one(data, group_col='mortality', target_name_survivor='Survivors', target_name_nonsurvivor='Non-survivors'):
    print(f"Creating Table 1, grouping by '{group_col}'...")
    
    table1_data = []
    n_total = len(data)
    group0_data = data[data[group_col] == 0]
    group1_data = data[data[group_col] == 1]
    n_group0 = len(group0_data)
    n_group1 = len(group1_data)

    table1_data.append(['Characteristic', 'Overall (N={})'.format(n_total), 
                        '{} (N={})'.format(target_name_survivor, n_group0), 
                        '{} (N={})'.format(target_name_nonsurvivor, n_group1), 'p-value'])
    table1_data.append(['N (%)', '-', f"{n_group0} ({n_group0/n_total*100:.1f}%)", f"{n_group1} ({n_group1/n_total*100:.1f}%)", ''])

    potential_features = [col for col in data.columns if col not in ['subject_id', 'hadm_id', 'stay_id', group_col, 'gender', 'deathtime', 'los_hours', 'intime', 'outtime']]
    
    for col in potential_features:
        if data[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            if data[col].nunique() > 10:
                overall_mean, overall_std = data[col].mean(), data[col].std()
                group0_mean, group0_std = group0_data[col].mean(), group0_data[col].std()
                group1_mean, group1_std = group1_data[col].mean(), group1_data[col].std()
                
                stat_val, p_val = stats.ttest_ind(group0_data[col].dropna(), group1_data[col].dropna(), equal_var=False)
                
                table1_data.append([col, f"{overall_mean:.2f} ± {overall_std:.2f}",
                                    f"{group0_mean:.2f} ± {group0_std:.2f}",
                                    f"{group1_mean:.2f} ± {group1_std:.2f}", f"{p_val:.3g}"])
            else:
                if data[col].nunique() <= 2 and (data[col].min()==0 and data[col].max()==1):
                    overall_n1 = data[col].sum()
                    group0_n1 = group0_data[col].sum()
                    group1_n1 = group1_data[col].sum()

                    overall_perc = overall_n1 / n_total * 100
                    group0_perc = group0_n1 / n_group0 * 100 if n_group0 > 0 else 0
                    group1_perc = group1_n1 / n_group1 * 100 if n_group1 > 0 else 0
                    
                    contingency_table = pd.crosstab(data[col], data[group_col])
                    if contingency_table.shape == (2,2) and contingency_table.values.min() >=0:
                        try:
                            chi2, p_val, _, _ = stats.chi2_contingency(contingency_table.applymap(lambda x: x if x>0 else 0.5))
                        except ValueError:
                            p_val = 1.0
                    else:
                        p_val = np.nan


                    table1_data.append([f"{col} (1, N (%))", f"{overall_n1} ({overall_perc:.1f}%)",
                                        f"{group0_n1} ({group0_perc:.1f}%)",
                                        f"{group1_n1} ({group1_perc:.1f}%)", f"{p_val:.3g}"])
    
    table1_df = pd.DataFrame(table1_data[1:], columns=table1_data[0])
    print(table1_df)
    table1_df.to_csv('table_one_mortality.csv', index=False)
    print("Table 1 saved to table_one_mortality.csv")
    return table1_df

def visualize_features(data, group_col='mortality', output_dir=FIGURES_PATH):
    print(f"Creating feature visualizations, grouped by '{group_col}'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    key_features_continuous = [
        'age', 'bmi', 
        'heart_rate_mean', 'respiratory_rate_mean', 'map_mean', 'temperature_mean', 'sbp_mean', 'dbp_mean', 'spo2_mean',
        'bun_mean', 'creatinine_mean', 'glucose_lab_mean', 'hemoglobin_mean', 'wbc_mean', 'lactate_mean', 'platelets_mean',
        'sodium_mean', 'potassium_mean', 'bicarbonate_mean', 'anion_gap_mean',
        'prev_dx_count_total'
    ]
    dynamic_cont_features = [col for col in data.columns if any(col.endswith(suffix) for suffix in ['_mean', '_min', '_max', '_std']) and col not in ['subject_id', 'hadm_id', 'stay_id', group_col]]
    dynamic_cont_features.extend(['age', 'bmi', 'prev_dx_count_total'])
    key_features_continuous = list(set(dynamic_cont_features))

    for feature in key_features_continuous:
        if feature in data.columns and data[feature].notna().sum() > 0:
            plt.figure(figsize=(10, 6))
            
            valid_data = data[[feature, group_col]].dropna()
            if valid_data.empty:
                print(f"Skipping {feature} visualization: no valid data after dropna.")
                plt.close()
                continue

            q_low = valid_data[feature].quantile(0.01)
            q_high = valid_data[feature].quantile(0.99)
            
            if q_low == q_high:
                plot_data = valid_data
            else:
                plot_data = valid_data[(valid_data[feature] >= q_low) & (valid_data[feature] <= q_high)]
            
            if plot_data.empty or plot_data[group_col].nunique() < 2:
                 sns.histplot(data=valid_data, x=feature, kde=True, hue=group_col if valid_data[group_col].nunique() >=2 else None, palette='viridis', multiple="stack" if valid_data[group_col].nunique() >=2 else "layer")
            else:
                 sns.histplot(data=plot_data, x=feature, kde=True, hue=group_col, palette='viridis', multiple="stack")


            plt.title(f'Distribution of {feature} by {group_col}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend(title=group_col, labels=['Survivor (0)', 'Non-Survivor (1)'] if group_col in plot_data else None)
            plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'), dpi=300)
            plt.close()
    print(f"Feature visualizations saved to {output_dir}")

if __name__ == "__main__":
    print("Starting TODO 2: Feature Extraction and Descriptive Analysis...")

    demographics_df = extract_demographics()
    bmi_df = extract_bmi()
    vitals_df = extract_vitals()
    labs_df = extract_labs()
    prev_diagnoses_df = extract_previous_diagnoses()

    features_df = final_cohort.copy()

    features_df = pd.merge(features_df, demographics_df.drop(columns=['gender']), on=['subject_id', 'hadm_id'], how='left')
    
    features_df = pd.merge(features_df, bmi_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    
    features_df = pd.merge(features_df, vitals_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    
    features_df = pd.merge(features_df, labs_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    
    features_df = pd.merge(features_df, prev_diagnoses_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')

    print(f"\nMerged features dataframe shape: {features_df.shape}")
    print(f"Number of unique stay_ids in features_df: {features_df['stay_id'].nunique()}")
    print(f"Columns in features_df: {features_df.columns.tolist()}")

    if 'mortality' not in features_df.columns:
        if 'deathtime' in features_df.columns:
             features_df['mortality'] = ~features_df['deathtime'].isna()
        else:
            print("ERROR: Mortality label not found in features_df.")
            exit()
    features_df['mortality'] = features_df['mortality'].astype(int)

    features_df.to_csv('features_for_mortality_prediction.csv', index=False)
    print("\nMerged features saved to 'features_for_mortality_prediction.csv'")

    table_one = create_table_one(features_df, group_col='mortality')
    
    visualize_features(features_df, group_col='mortality')

    print("\nTODO 2 processing complete.")