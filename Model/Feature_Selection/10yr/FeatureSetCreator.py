from lifelines import CoxPHFitter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_initial = pd.read_csv('../../Files/10yr/Train_Preprocessed_Data.csv')

data = data_initial.drop(
    [ 'tnm.t',
     'tnm.n', 'tnm.m',
     'tp53_mutation'
     ], axis=1)
# 'tnm.m', 'cin_status', 'rfs_months_censored_5yr', 'sex', 'age_at_diagnosis_in_years', 'tnm_stage', 'tnm.t',
# 'tnm.n', 'tnm.m', 'tumour_location', 'chemotherapy_adjuvant', 'mmr_status', 'cimp_status', 'cin_status',
# 'tp53_mutation', 'kras_mutation', 'braf_mutation'    'CMS', 'os_months_censored_10yr', 'os_event_censored_10yr',
# 'rfs_months_censored_10yr', 'rfs_event_censored_10yr'
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables to dummy variables

data.to_csv('../../Files/10yr/FeatureSets/Best_Features_10.csv', index=False)
