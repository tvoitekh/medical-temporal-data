import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re

plt.style.use('ggplot')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

BASE_PATH = './Data/'
ICU_PATH = BASE_PATH + 'icu/'
HOSP_PATH = BASE_PATH + 'hosp/'
LABEL_PATH = BASE_PATH + 'label/'

def load_icu_stays():
    print("Loading ICU stays data...")
    return pd.read_csv(ICU_PATH + '_icustays.csv')

def load_patients():
    print("Loading patients data...")
    return pd.read_csv(HOSP_PATH + '_patients.csv')

def load_admissions():
    print("Loading admissions data...")
    return pd.read_csv(HOSP_PATH + '_admissions.csv')

def load_mortality_labels():
    print("Loading mortality labels...")
    return pd.read_csv(LABEL_PATH + '_label_death.csv')

def calculate_hours_diff(intime, outtime):
    try:
        in_time = datetime.strptime(intime, '%Y-%m-%d %H:%M:%S')
        out_time = datetime.strptime(outtime, '%Y-%m-%d %H:%M:%S')
        return (out_time - in_time).total_seconds() / 3600
    except:
        return None

def select_study_cohort():
    print("Selecting study cohort...")
    
    icu_stays = load_icu_stays()
    patients = load_patients()
    admissions = load_admissions()
    mortality_labels = load_mortality_labels()
    
    total_patients = patients.shape[0]
    total_icu_stays = icu_stays.shape[0]
    print(f"Initial patients count: {total_patients}")
    print(f"Initial ICU stays count: {total_icu_stays}")
    
    patients_with_icu = patients[patients['subject_id'].isin(icu_stays['subject_id'])].copy()
    print(f"Patients with at least one ICU stay: {patients_with_icu.shape[0]}")
    
    icu_stays['los_hours'] = icu_stays.apply(
        lambda x: calculate_hours_diff(x['intime'], x['outtime']), axis=1)
    
    icu_stays = icu_stays.sort_values(['subject_id', 'intime'])
    first_icu_stays = icu_stays.groupby('subject_id').first().reset_index()
    print(f"First ICU stays count: {first_icu_stays.shape[0]}")
    
    first_icu_stays_6h = first_icu_stays[first_icu_stays['los_hours'] >= 6].copy()
    print(f"First ICU stays with at least 6 hours: {first_icu_stays_6h.shape[0]}")
    
    final_cohort = pd.merge(
        first_icu_stays_6h, 
        mortality_labels,
        on=['subject_id', 'hadm_id', 'stay_id'],
        how='left'
    )
    
    final_cohort['mortality'] = ~final_cohort['deathtime'].isna()
    print(f"Final cohort size: {final_cohort.shape[0]}")
    print(f"Mortality rate in cohort: {final_cohort['mortality'].mean():.2%}")
    
    flowchart_data = {
        'Initial patients': total_patients,
        'Patients with ICU stay': patients_with_icu.shape[0],
        'First ICU stay only': first_icu_stays.shape[0],
        'ICU stay ≥ 6 hours': first_icu_stays_6h.shape[0],
        'Final cohort': final_cohort.shape[0]
    }
    
    survivors = final_cohort[~final_cohort['mortality']].shape[0]
    non_survivors = final_cohort[final_cohort['mortality']].shape[0]
    print(f"Survivors: {survivors} ({survivors/final_cohort.shape[0]:.2%})")
    print(f"Non-survivors: {non_survivors} ({non_survivors/final_cohort.shape[0]:.2%})")
    
    return final_cohort, flowchart_data

def plot_cohort_flowchart(flowchart_data):
    plt.figure(figsize=(10, 8))
    
    keys = list(flowchart_data.keys())
    values = list(flowchart_data.values())
    
    excluded = []
    for i in range(len(values) - 1):
        excluded.append(values[i] - values[i+1])
    excluded.append(0)
    
    y_pos = np.arange(len(keys))
    plt.barh(y_pos, values, align='center', alpha=0.7)
    
    for i, (v, e) in enumerate(zip(values, excluded)):
        plt.text(v + 10, i, f"{v}", va='center')
        if e > 0:
            plt.text(v - 200, i - 0.4, f"Excluded: {e}", va='center', color='red', fontsize=9)
    
    plt.yticks(y_pos, keys)
    plt.xlabel('Number of Patients')
    plt.title('Cohort Selection Flowchart')
    
    for i in range(len(keys) - 1):
        plt.annotate('', xy=(0, i+0.9), xytext=(0, i+0.1),
                    arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('cohort_flowchart.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    final_cohort, flowchart_data = select_study_cohort()
    plot_cohort_flowchart(flowchart_data)
    
    final_cohort.to_csv('final_cohort.csv', index=False)
    print("Cohort selection complete and saved to 'final_cohort.csv'")