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
    final_cohort = pd.read_csv('final_cohort_task2.csv')
    print(f"Loaded cohort with {final_cohort.shape[0]} stays from final_cohort_task2.csv")
except FileNotFoundError:
    print("final_cohort_task2.csv not found. Please run the cohort selection script (TODO 1) first.")
    exit()

def load_labevents():
    print("Loading lab events data...")
    df = pd.read_csv(HOSP_PATH + '_labevents.csv')
    return df

def load_diagnoses():
    print("Loading diagnoses data...")
    df = pd.read_csv(HOSP_PATH + '_diagnoses_icd.csv')
    return df

def load_chartevents_data():
    print("Loading chart events data...")
    df = pd.read_csv(ICU_PATH + '_chartevents.csv')
    return df

def calculate_bmi(height_cm, weight_kg):
    if pd.isna(height_cm) or pd.isna(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return np.nan
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)

def extract_demographics():
    print("Extracting demographics...")
    patients = pd.read_csv(HOSP_PATH + '_patients.csv')
    cohort_subject_ids = final_cohort['subject_id'].unique()
    cohort_patients = patients[patients['subject_id'].isin(cohort_subject_ids)].copy()

    gender_map = {'M': 1, 'F': 0}
    cohort_patients['gender_numeric'] = cohort_patients['gender'].map(gender_map)

    admissions = pd.read_csv(HOSP_PATH + '_admissions.csv')
    cohort_hadm_ids = final_cohort['hadm_id'].unique()
    cohort_admissions = admissions[admissions['hadm_id'].isin(cohort_hadm_ids)].copy()

    demographics = pd.merge(
        cohort_patients[['subject_id', 'gender', 'gender_numeric', 'anchor_year', 'anchor_age']],
        cohort_admissions[['subject_id', 'hadm_id', 'admittime']],
        on='subject_id',
        how='inner'
    )
    demographics = demographics[demographics['hadm_id'].isin(cohort_hadm_ids)]

    demographics['admittime'] = pd.to_datetime(demographics['admittime'])
    demographics['admittime_year'] = demographics['admittime'].dt.year
    demographics['age'] = demographics['anchor_age'] + (demographics['admittime_year'] - demographics['anchor_year'])

    demographics.loc[demographics['age'] > 89, 'age'] = 90

    demographics = demographics[['subject_id', 'hadm_id', 'gender_numeric', 'age']].drop_duplicates()

    print(f"Extracted demographics for {demographics['subject_id'].nunique()} unique subjects, {demographics['hadm_id'].nunique()} unique hospital admissions.")
    return demographics

def extract_bmi(chartevents_df, cohort_df_for_bmi):
    print("Extracting BMI...")

    cohort_stays_info = cohort_df_for_bmi[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    cohort_stays_info['intime'] = pd.to_datetime(cohort_stays_info['intime'])
    cohort_stays_info['bmi_cutoff_time'] = cohort_stays_info['intime'] + pd.Timedelta(hours=24)

    chartevents_for_bmi_calc = chartevents_df[chartevents_df['stay_id'].isin(cohort_stays_info['stay_id'])].copy()
    if chartevents_for_bmi_calc.empty:
        print("No chartevents records found for the stay_ids in the cohort for BMI.")
        return pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'bmi'])

    chartevents_for_bmi_calc['charttime'] = pd.to_datetime(chartevents_for_bmi_calc['charttime'])

    chartevents_for_bmi_calc = pd.merge(
        chartevents_for_bmi_calc,
        cohort_stays_info[['stay_id', 'intime', 'bmi_cutoff_time']],
        on='stay_id',
        how='left'
    )
    chartevents_for_bmi_calc.dropna(subset=['intime', 'bmi_cutoff_time'], inplace=True)

    chartevents_bmi_window = chartevents_for_bmi_calc[
        (chartevents_for_bmi_calc['charttime'] >= chartevents_for_bmi_calc['intime']) &
        (chartevents_for_bmi_calc['charttime'] <= chartevents_for_bmi_calc['bmi_cutoff_time'])
    ].copy()

    if chartevents_bmi_window.empty:
        print("No chartevents records found within the 24-hour BMI window.")
        return pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'bmi'])

    height_itemids_cm = [226730]
    height_itemids_in = [226707]
    weight_itemids_kg = [226512, 224639]
    weight_itemids_lbs = [226531]

    all_height_itemids = height_itemids_cm + height_itemids_in
    all_weight_itemids = weight_itemids_kg + weight_itemids_lbs

    heights_raw_df = chartevents_bmi_window[chartevents_bmi_window['itemid'].isin(all_height_itemids)].copy()
    weights_raw_df = chartevents_bmi_window[chartevents_bmi_window['itemid'].isin(all_weight_itemids)].copy()

    processed_heights_list = []
    if not heights_raw_df.empty:
        for itemid_cm in height_itemids_cm:
            temp_h_cm = heights_raw_df[heights_raw_df['itemid'] == itemid_cm].copy()
            if not temp_h_cm.empty:
                temp_h_cm.rename(columns={'valuenum': 'height_cm'}, inplace=True)
                processed_heights_list.append(temp_h_cm[['stay_id', 'subject_id', 'hadm_id', 'charttime', 'height_cm']])
        for itemid_in in height_itemids_in:
            temp_h_in = heights_raw_df[heights_raw_df['itemid'] == itemid_in].copy()
            if not temp_h_in.empty:
                temp_h_in['height_cm'] = temp_h_in['valuenum'] * 2.54
                processed_heights_list.append(temp_h_in[['stay_id', 'subject_id', 'hadm_id', 'charttime', 'height_cm']])

    all_heights_converted_df = pd.DataFrame()
    if processed_heights_list:
        all_heights_converted_df = pd.concat(processed_heights_list)
        all_heights_converted_df = all_heights_converted_df[all_heights_converted_df['height_cm'].notna() & (all_heights_converted_df['height_cm'] > 50) & (all_heights_converted_df['height_cm'] < 250)]

    processed_weights_list = []
    if not weights_raw_df.empty:
        for itemid_kg in weight_itemids_kg:
            temp_w_kg = weights_raw_df[weights_raw_df['itemid'] == itemid_kg].copy()
            if not temp_w_kg.empty:
                temp_w_kg.rename(columns={'valuenum': 'weight_kg'}, inplace=True)
                processed_weights_list.append(temp_w_kg[['stay_id', 'subject_id', 'hadm_id', 'charttime', 'weight_kg']])
        for itemid_lbs in weight_itemids_lbs:
            temp_w_lbs = weights_raw_df[weights_raw_df['itemid'] == itemid_lbs].copy()
            if not temp_w_lbs.empty:
                temp_w_lbs['weight_kg'] = temp_w_lbs['valuenum'] * 0.453592
                processed_weights_list.append(temp_w_lbs[['stay_id', 'subject_id', 'hadm_id', 'charttime', 'weight_kg']])

    all_weights_converted_df = pd.DataFrame()
    if processed_weights_list:
        all_weights_converted_df = pd.concat(processed_weights_list)
        all_weights_converted_df = all_weights_converted_df[all_weights_converted_df['weight_kg'].notna() & (all_weights_converted_df['weight_kg'] > 20) & (all_weights_converted_df['weight_kg'] < 300)]

    if all_heights_converted_df.empty or all_weights_converted_df.empty:
        print("No valid height or weight events found after processing units and plausible value filtering.")
        return pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'bmi'])

    earliest_valid_heights = all_heights_converted_df.sort_values(['stay_id', 'charttime']).groupby('stay_id').first().reset_index()
    earliest_valid_weights = all_weights_converted_df.sort_values(['stay_id', 'charttime']).groupby('stay_id').first().reset_index()

    bmi_calculation_base_df = pd.merge(
        earliest_valid_heights[['subject_id', 'hadm_id', 'stay_id', 'height_cm']],
        earliest_valid_weights[['stay_id', 'weight_kg']],
        on='stay_id',
        how='inner'
    )

    if bmi_calculation_base_df.empty:
        print("No stays found with both valid height and weight measurements.")
        return pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'bmi'])

    bmi_calculation_base_df['bmi'] = bmi_calculation_base_df.apply(
        lambda x: calculate_bmi(x['height_cm'], x['weight_kg']), axis=1
    )

    final_bmi_df = bmi_calculation_base_df[(bmi_calculation_base_df['bmi'] >= 10) & (bmi_calculation_base_df['bmi'] <= 70)]
    final_bmi_df = final_bmi_df[['subject_id', 'hadm_id', 'stay_id', 'bmi']].drop_duplicates()

    print(f"Extracted BMI for {final_bmi_df['stay_id'].nunique()} ICU stays.")
    return final_bmi_df

def extract_vitals(chartevents_df, cohort_df_for_vitals):
    print("Extracting vital signs...")

    cohort_chartevents = chartevents_df[chartevents_df['stay_id'].isin(cohort_df_for_vitals['stay_id'])].copy()
    if cohort_chartevents.empty:
        print("No chartevents records found for the stay_ids in the cohort for Vitals.")
        return cohort_df_for_vitals[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    cohort_chartevents['charttime'] = pd.to_datetime(cohort_chartevents['charttime'])

    icu_stay_times = cohort_df_for_vitals[['stay_id', 'intime']].copy()
    icu_stay_times['intime'] = pd.to_datetime(icu_stay_times['intime'])
    icu_stay_times['vitals_cutoff_time'] = icu_stay_times['intime'] + pd.Timedelta(hours=6)

    cohort_chartevents = pd.merge(cohort_chartevents, icu_stay_times, on='stay_id', how='left')
    cohort_chartevents.dropna(subset=['intime', 'vitals_cutoff_time'], inplace=True)

    vitals_first_6h = cohort_chartevents[
        (cohort_chartevents['charttime'] >= cohort_chartevents['intime']) &
        (cohort_chartevents['charttime'] <= cohort_chartevents['vitals_cutoff_time'])
    ].copy()

    if vitals_first_6h.empty:
        print("No vital sign chartevents records found within the first 6 hours.")
        return cohort_df_for_vitals[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    vital_itemids_map = {
        220045: 'heart_rate', 220210: 'respiratory_rate', 220052: 'map',
        223761: 'temperature', 223762: 'temperature_f_raw',
        220179: 'sbp', 220180: 'dbp', 220277: 'spo2'
    }

    vitals_processed = vitals_first_6h[vitals_first_6h['itemid'].isin(vital_itemids_map.keys())].copy()
    vitals_processed['vital_name'] = vitals_processed['itemid'].map(vital_itemids_map)

    f_temp_mask = vitals_processed['vital_name'] == 'temperature_f_raw'
    if f_temp_mask.any():
        vitals_processed.loc[f_temp_mask, 'valuenum'] = (vitals_processed.loc[f_temp_mask, 'valuenum'] - 32) * 5/9
        vitals_processed.loc[f_temp_mask, 'vital_name'] = 'temperature'

    vitals_processed = vitals_processed[vitals_processed['vital_name'] != 'temperature_f_raw']

    aggregated_vitals_list = []
    unique_vital_names = sorted(list(vitals_processed['vital_name'].dropna().unique()))

    for name in unique_vital_names:
        vital_subset = vitals_processed[vitals_processed['vital_name'] == name]
        if not vital_subset.empty:
            agg_functions = ['mean', 'min', 'max', 'std', 'count']
            summary_df = vital_subset.groupby(['subject_id', 'hadm_id', 'stay_id'])['valuenum'].agg(agg_functions).reset_index()
            summary_df.columns = ['subject_id', 'hadm_id', 'stay_id'] + [f'{name}_{func}' for func in agg_functions]
            aggregated_vitals_list.append(summary_df)

    if not aggregated_vitals_list:
        print("No vital signs data found for aggregation.")
        return cohort_df_for_vitals[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    vital_features_df = cohort_df_for_vitals[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
    for agg_df in aggregated_vitals_list:
        vital_features_df = pd.merge(vital_features_df, agg_df, on=['subject_id', 'hadm_id', 'stay_id'], how='left')

    print(f"Extracted vital signs aggregates for {vital_features_df['stay_id'].nunique()} ICU stays.")
    return vital_features_df

def extract_labs():
    print("Extracting laboratory results...")
    labevents = load_labevents()

    cohort_labevents = labevents[labevents['hadm_id'].isin(final_cohort['hadm_id'])].copy()
    if cohort_labevents.empty:
        print("No labevent records found for the hadm_ids in the cohort.")
        return final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    cohort_labevents['charttime'] = pd.to_datetime(cohort_labevents['charttime'])

    lab_itemids_map = {
        51006: 'bun', 50863: 'alkaline_phosphatase', 50885: 'bilirubin_total',
        50912: 'creatinine', 50931: 'glucose_lab', 51265: 'platelets',
        51222: 'hemoglobin', 51301: 'wbc', 50824: 'sodium',
        50971: 'potassium', 50809: 'glucose_bg', 50818: 'lactate',
        51221: 'hematocrit', 50902: 'chloride', 50882: 'bicarbonate', 50868: 'anion_gap'
    }

    labs_processed = cohort_labevents[cohort_labevents['itemid'].isin(lab_itemids_map.keys())].copy()
    labs_processed['lab_name'] = labs_processed['itemid'].map(lab_itemids_map)

    icu_stay_times = final_cohort[['hadm_id', 'stay_id', 'intime']].drop_duplicates(subset=['hadm_id', 'stay_id'])
    icu_stay_times['intime'] = pd.to_datetime(icu_stay_times['intime'])
    icu_stay_times['labs_cutoff_time'] = icu_stay_times['intime'] + pd.Timedelta(hours=6)

    labs_processed = pd.merge(labs_processed, icu_stay_times, on='hadm_id', how='left')
    labs_processed.dropna(subset=['stay_id', 'intime', 'labs_cutoff_time'], inplace=True)

    labs_first_6h = labs_processed[
        (labs_processed['charttime'] >= labs_processed['intime']) &
        (labs_processed['charttime'] <= labs_processed['labs_cutoff_time'])
    ].copy()

    if labs_first_6h.empty:
        print("No lab records found within the first 6 hours.")
        return final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    aggregated_labs_list = []
    unique_lab_names = sorted(list(labs_first_6h['lab_name'].dropna().unique()))

    for name in unique_lab_names:
        lab_subset = labs_first_6h[labs_first_6h['lab_name'] == name]
        if not lab_subset.empty:
            agg_functions = ['mean', 'min', 'max', 'std', 'count', 'first', 'last']
            summary_df = lab_subset.groupby(['subject_id', 'hadm_id', 'stay_id'])['valuenum'].agg(agg_functions).reset_index()

            new_colnames = ['subject_id', 'hadm_id', 'stay_id'] + [f'{name}_{func}' for func in agg_functions]
            summary_df.columns = new_colnames

            if f'{name}_first' in summary_df.columns and f'{name}_last' in summary_df.columns:
                summary_df[f'{name}_delta'] = summary_df[f'{name}_last'] - summary_df[f'{name}_first']

            aggregated_labs_list.append(summary_df)

    if not aggregated_labs_list:
        print("No lab results data found for aggregation.")
        return final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    lab_features_df = final_cohort[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
    for agg_df in aggregated_labs_list:
        lab_features_df = pd.merge(lab_features_df, agg_df, on=['subject_id', 'hadm_id', 'stay_id'], how='left')

    print(f"Extracted lab results aggregates for {lab_features_df['stay_id'].nunique()} ICU stays.")
    return lab_features_df

def extract_previous_diagnoses():
    print("Extracting previous diagnoses...")
    diagnoses_icd_df = load_diagnoses()
    admissions_df = pd.read_csv(HOSP_PATH + '_admissions.csv')

    current_stays_info = final_cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    current_stays_info['intime'] = pd.to_datetime(current_stays_info['intime'])

    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])

    all_prev_dx_dfs_list = []

    for _, current_stay_row in current_stays_info.iterrows():
        subj_id = current_stay_row['subject_id']
        curr_icu_intime = current_stay_row['intime']

        subject_admissions = admissions_df[admissions_df['subject_id'] == subj_id].sort_values(by='admittime')
        prev_hospital_admissions = subject_admissions[subject_admissions['admittime'] < curr_icu_intime]

        if not prev_hospital_admissions.empty:
            prev_hadm_ids_list = prev_hospital_admissions['hadm_id'].unique()
            diagnoses_for_prev_adms = diagnoses_icd_df[diagnoses_icd_df['hadm_id'].isin(prev_hadm_ids_list)].copy()

            if not diagnoses_for_prev_adms.empty:
                diagnoses_for_prev_adms['stay_id'] = current_stay_row['stay_id']
                all_prev_dx_dfs_list.append(diagnoses_for_prev_adms)

    if not all_prev_dx_dfs_list:
        print("No previous hospitalizations with diagnoses found for any patient in the cohort based on the provided _admissions.csv.")
        prev_dx_placeholder = final_cohort[['subject_id','hadm_id','stay_id']].drop_duplicates().copy()
        prev_dx_placeholder['has_previous_admission_with_dx'] = 0
        prev_dx_placeholder['prev_dx_count_total'] = 0
        return prev_dx_placeholder

    combined_prev_dx = pd.concat(all_prev_dx_dfs_list)

    def map_icd9_to_chapter_simplified(icd_code_str):
        if pd.isna(icd_code_str): return 'unknown'
        icd_code_str = str(icd_code_str).upper()
        if any(prefix in icd_code_str for prefix in ['V', 'E']): return 'other_icd9'
        try:
            numeric_part = int(re.match(r"([0-9]+)", icd_code_str).group(1))
            if 1 <= numeric_part <= 139: return 'infectious_parasitic'
            if 140 <= numeric_part <= 239: return 'neoplasms'
            if 240 <= numeric_part <= 279: return 'endocrine_metabolic'
            if 290 <= numeric_part <= 319: return 'mental_disorders'
            if 320 <= numeric_part <= 389: return 'nervous_sensory'
            if 390 <= numeric_part <= 459: return 'circulatory'
            if 460 <= numeric_part <= 519: return 'respiratory'
            if 520 <= numeric_part <= 579: return 'digestive'
            if 580 <= numeric_part <= 629: return 'genitourinary'
            if 680 <= numeric_part <= 709: return 'skin_subcutaneous'
            if 710 <= numeric_part <= 739: return 'musculoskeletal_connective'
            if 780 <= numeric_part <= 799: return 'symptoms_signs_illdefined'
            if 800 <= numeric_part <= 999: return 'injury_poisoning'
            return 'other_icd9'
        except: return 'other_icd9'

    def map_icd10_to_chapter_simplified(icd_code_str):
        if pd.isna(icd_code_str): return 'unknown'
        icd_code_str = str(icd_code_str).upper()
        if 'A' <= icd_code_str[0] <= 'B': return 'infectious_parasitic'
        if 'C' <= icd_code_str[0] <= 'D' and (len(icd_code_str) > 1 and '0' <= icd_code_str[1] <= '4'): return 'neoplasms'
        if icd_code_str.startswith('E'): return 'endocrine_metabolic'
        if icd_code_str.startswith('F'): return 'mental_disorders'
        if icd_code_str.startswith('G'): return 'nervous_sensory'
        if icd_code_str.startswith('I'): return 'circulatory'
        if icd_code_str.startswith('J'): return 'respiratory'
        if icd_code_str.startswith('K'): return 'digestive'
        if icd_code_str.startswith('N'): return 'genitourinary'
        if icd_code_str.startswith('L'): return 'skin_subcutaneous'
        if icd_code_str.startswith('M'): return 'musculoskeletal_connective'
        if icd_code_str.startswith('R'): return 'symptoms_signs_illdefined'
        if 'S' <= icd_code_str[0] <= 'T': return 'injury_poisoning'
        return 'other_icd10'

    combined_prev_dx['icd_chapter'] = combined_prev_dx.apply(
        lambda r: map_icd9_to_chapter_simplified(r['icd_code']) if r['icd_version'] == 9
        else map_icd10_to_chapter_simplified(r['icd_code']) if r['icd_version'] == 10
        else 'unknown_version', axis=1
    )

    dx_total_counts = combined_prev_dx.groupby('stay_id').size().reset_index(name='prev_dx_count_total')
    dx_chapter_counts = combined_prev_dx.groupby(['stay_id', 'icd_chapter']).size().unstack(fill_value=0)
    dx_chapter_counts.columns = [f'prev_dx_{col}_count' for col in dx_chapter_counts.columns]
    dx_chapter_counts.reset_index(inplace=True)

    prev_dx_features_df = pd.merge(final_cohort[['subject_id','hadm_id','stay_id']].drop_duplicates(), dx_total_counts, on='stay_id', how='left')
    prev_dx_features_df = pd.merge(prev_dx_features_df, dx_chapter_counts, on='stay_id', how='left')

    prev_dx_features_df['has_previous_admission_with_dx'] = np.where(prev_dx_features_df['prev_dx_count_total'].notna() & (prev_dx_features_df['prev_dx_count_total'] > 0), 1, 0)
    prev_dx_features_df.fillna(0, inplace=True)

    print(f"Extracted previous diagnoses features for {prev_dx_features_df['stay_id'].nunique()} ICU stays.")
    return prev_dx_features_df

def create_table_one(data, group_col='mortality', target_name_survivor='Survivors (0)', target_name_nonsurvivor='Non-survivors (1)'):
    print(f"Creating Table 1, grouping by '{group_col}'...")

    table_one_rows = []
    n_total_overall = len(data)
    group0_subset = data[data[group_col] == 0]
    group1_subset = data[data[group_col] == 1]
    n_group0_count = len(group0_subset)
    n_group1_count = len(group1_subset)

    table_one_rows.append(['Characteristic', f'Overall (N={n_total_overall})',
                           f'{target_name_survivor} (N={n_group0_count})',
                           f'{target_name_nonsurvivor} (N={n_group1_count})', 'p-value'])
    table_one_rows.append(['N (%)', '-', f"{n_group0_count} ({n_group0_count/n_total_overall*100:.1f}%)",
                           f"{n_group1_count} ({n_group1_count/n_total_overall*100:.1f}%)", ''])

    cols_for_table1 = [col for col in data.columns if col not in ['subject_id', 'hadm_id', 'stay_id', group_col, 'gender', 'deathtime', 'los_hours', 'intime', 'outtime', 'anchor_year', 'admittime', 'admittime_year', 'hospital_expire_flag'] + [c for c in data.columns if c.endswith(('_first', '_last', '_id', '_text', '_time', '_date'))]]

    for feature_col in sorted(cols_for_table1):
        if data[feature_col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            is_binary_like = (data[feature_col].min()==0 and data[feature_col].max()==1 and data[feature_col].nunique()<=2)
            if data[feature_col].nunique() > 5 and not is_binary_like:
                mean_overall, std_overall = data[feature_col].mean(), data[feature_col].std()
                mean_g0, std_g0 = group0_subset[feature_col].mean(), group0_subset[feature_col].std()
                mean_g1, std_g1 = group1_subset[feature_col].mean(), group1_subset[feature_col].std()

                stat, pval = stats.ttest_ind(group0_subset[feature_col].dropna(), group1_subset[feature_col].dropna(), equal_var=False, nan_policy='omit')

                table_one_rows.append([feature_col, f"{mean_overall:.2f} ± {std_overall:.2f}",
                                       f"{mean_g0:.2f} ± {std_g0:.2f}",
                                       f"{mean_g1:.2f} ± {std_g1:.2f}", f"{pval:.3g}" if not pd.isna(pval) else "N/A"])
            else:
                val_for_perc = 1 if is_binary_like else data[feature_col].value_counts().index[0] if not data[feature_col].empty and not data[feature_col].value_counts().empty else 'N/A'
                count_overall_cat_val = data[data[feature_col] == val_for_perc].shape[0] if val_for_perc != 'N/A' else 0
                perc_overall = count_overall_cat_val / n_total_overall * 100 if n_total_overall > 0 else 0

                count_g0_cat_val = group0_subset[group0_subset[feature_col] == val_for_perc].shape[0] if val_for_perc != 'N/A' else 0
                perc_g0 = count_g0_cat_val / n_group0_count * 100 if n_group0_count > 0 else 0

                count_g1_cat_val = group1_subset[group1_subset[feature_col] == val_for_perc].shape[0] if val_for_perc != 'N/A' else 0
                perc_g1 = count_g1_cat_val / n_group1_count * 100 if n_group1_count > 0 else 0

                pval_cat = "N/A"
                try:
                    if data[feature_col].nunique() <=2 and data[group_col].nunique() <=2 and val_for_perc != 'N/A':
                        contingency = pd.crosstab(data[feature_col].fillna(-1), data[group_col].fillna(-1))
                        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                             chi2_stat, pval_cat_val, _, _ = stats.chi2_contingency(contingency)
                             pval_cat = f"{pval_cat_val:.3g}"
                except Exception:
                    pval_cat = "Error"

                table_one_rows.append([f"{feature_col} ({val_for_perc}, N (%))", f"{count_overall_cat_val} ({perc_overall:.1f}%)",
                                       f"{count_g0_cat_val} ({perc_g0:.1f}%)",
                                       f"{count_g1_cat_val} ({perc_g1:.1f}%)", pval_cat])

    table_one_dataframe = pd.DataFrame(table_one_rows[1:], columns=table_one_rows[0])
    print(table_one_dataframe)
    table_one_dataframe.to_csv(os.path.join(FIGURES_PATH, 'table_one_mortality.csv'), index=False)
    print(f"Table 1 saved to {os.path.join(FIGURES_PATH, 'table_one_mortality.csv')}")
    return table_one_dataframe

def visualize_selected_features(data, group_col='mortality', output_dir=FIGURES_PATH):
    print(f"Creating selected feature visualizations, grouped by '{group_col}'...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    demographic_features = ['age', 'bmi']
    vital_means_to_plot = ['heart_rate_mean', 'respiratory_rate_mean', 'map_mean', 'temperature_mean', 'sbp_mean', 'spo2_mean']
    lab_means_to_plot = ['bun_mean', 'creatinine_mean', 'glucose_lab_mean', 'platelets_mean', 'hemoglobin_mean', 'wbc_mean', 'lactate_mean']
    prev_dx_features_to_plot = ['prev_dx_count_total', 'has_previous_admission_with_dx']

    def plot_hist_or_bar(feature_name, df_subset, group_identifier, is_categorical_like=False):
        plt.figure(figsize=(8, 5))
        valid_df = df_subset[[feature_name, group_identifier]].dropna()
        if valid_df.empty :
            plt.close()
            return

        if is_categorical_like or valid_df[feature_name].nunique() <= 5 :
             sns.countplot(data=valid_df, x=feature_name, hue=group_identifier, palette='viridis')
             if feature_name == 'gender_numeric': plt.xticks([0,1],['Female','Male'])
             elif feature_name == 'has_previous_admission_with_dx': plt.xticks([0,1],['No','Yes'])
        else:
            q_low = valid_df[feature_name].quantile(0.01)
            q_high = valid_df[feature_name].quantile(0.99)
            plot_df_subset = valid_df[(valid_df[feature_name] >= q_low) & (valid_df[feature_name] <= q_high)] if q_low != q_high else valid_df
            if plot_df_subset.empty: plot_df_subset = valid_df

            sns.histplot(data=plot_df_subset, x=feature_name, hue=group_identifier, kde=True, palette='viridis', multiple="stack" if plot_df_subset[group_identifier].nunique() > 1 else "layer")
            plt.ylabel('Frequency')

        plt.title(f'Distribution of {feature_name} by {group_identifier}')
        plt.xlabel(feature_name)
        if valid_df[group_identifier].nunique() > 1:
             handles, labels = plt.gca().get_legend_handles_labels()
             if len(labels) > 0 and len(handles) >0: # Ensure legend is possible
                try: # Attempt to set specific labels if mortality
                    if group_identifier == 'mortality' and len(handles) >=2 :
                        plt.legend(handles, ['Survivor (0)', 'Non-Survivor (1)'], title=group_identifier)
                    else:
                        plt.legend(title=group_identifier)
                except Exception: # Fallback default legend
                    if plt.gca().get_legend() is not None: plt.legend(title=group_identifier)


        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_name}_distribution.png'), dpi=200)
        plt.close()

    for feat in demographic_features + [prev_dx_features_to_plot[0]]:
        if feat in data.columns: plot_hist_or_bar(feat, data, group_col)

    if 'gender_numeric' in data.columns:
        plot_hist_or_bar('gender_numeric', data, group_col, is_categorical_like=True)
    if prev_dx_features_to_plot[1] in data.columns:
        plot_hist_or_bar(prev_dx_features_to_plot[1], data, group_col, is_categorical_like=True)

    def plot_grouped_boxplots(feature_list_names, title_prefix, filename_suffix, df, group):
        actual_features_present = [f for f in feature_list_names if f in df.columns and df[f].notna().sum() > 0]
        if not actual_features_present: return

        num_actual_features = len(actual_features_present)
        cols_per_fig = 3 ; rows_per_fig = (num_actual_features + cols_per_fig -1) // cols_per_fig
        plt.figure(figsize=(5 * cols_per_fig, 4 * rows_per_fig))
        for i, feature_name_item in enumerate(actual_features_present):
            plt.subplot(rows_per_fig, cols_per_fig, i + 1)
            sns.boxplot(x=group, y=feature_name_item, data=df, palette='viridis', showfliers=False)
            plt.title(feature_name_item.replace('_mean',' Mean')); plt.xlabel(''); plt.ylabel('')
        plt.suptitle(f'{title_prefix} (First 6 Hours) by {group}', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(os.path.join(output_dir, f'{filename_suffix}_boxplots.png'), dpi=200)
        plt.close()

    plot_grouped_boxplots(vital_means_to_plot, 'Key Vital Signs Means', 'vitals_means', data, group_col)
    plot_grouped_boxplots(lab_means_to_plot, 'Key Lab Results Means', 'labs_means', data, group_col)

    print(f"Selected feature visualizations saved to {output_dir}")


if __name__ == "__main__":
    print("Starting TODO 2: Feature Extraction and Descriptive Analysis...")

    chartevents_master_df = load_chartevents_data()

    demographics_df = extract_demographics()
    bmi_df = extract_bmi(chartevents_master_df, final_cohort)
    vitals_features_df = extract_vitals(chartevents_master_df, final_cohort)
    labs_features_df = extract_labs()
    previous_diagnoses_features_df = extract_previous_diagnoses()

    master_features_df = final_cohort.copy()

    master_features_df = pd.merge(master_features_df, demographics_df, on=['subject_id', 'hadm_id'], how='left')
    master_features_df = pd.merge(master_features_df, bmi_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    master_features_df = pd.merge(master_features_df, vitals_features_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    master_features_df = pd.merge(master_features_df, labs_features_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')
    master_features_df = pd.merge(master_features_df, previous_diagnoses_features_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore'), on='stay_id', how='left')

    print(f"\nMerged features dataframe shape: {master_features_df.shape}")
    print(f"Number of unique stay_ids in master_features_df: {master_features_df['stay_id'].nunique()}")

    if 'in_hospital_mortality' in master_features_df.columns:
         master_features_df.rename(columns={'in_hospital_mortality': 'mortality'}, inplace=True)
    elif 'mortality' not in master_features_df.columns:
        if 'deathtime' in master_features_df.columns:
             master_features_df['mortality'] = master_features_df['deathtime'].notna().astype(int)
        else:
            print("ERROR: Mortality label ('mortality' or 'deathtime') not found in master_features_df.")
            exit()
    master_features_df['mortality'] = master_features_df['mortality'].astype(int)


    master_features_df.to_csv('features_for_mortality_prediction.csv', index=False)
    print("\nMerged features saved to 'features_for_mortality_prediction.csv'")

    table1_df_output = create_table_one(master_features_df, group_col='mortality')

    visualize_selected_features(master_features_df, group_col='mortality')

    print("\nTODO 2 processing complete.")