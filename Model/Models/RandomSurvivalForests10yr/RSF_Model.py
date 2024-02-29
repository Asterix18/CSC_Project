import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from joblib import dump
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

data = pd.read_csv('../../Files/5yr/FeatureSets/Best_Features_6.csv')
data['os_event_censored_5yr'] = data['os_event_censored_5yr'].astype(bool)
features = data.drop(['os_event_censored_5yr', 'os_months_censored_5yr', 'rfs_months_censored_5yr'], axis=1)
time_to_event_data = data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)
feature_names = features.columns.tolist()
print(feature_names)
rsf = RandomSurvivalForest(max_depth=3, max_features=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100,
                           random_state=40)
rsf.fit(features, time_to_event_data)
print(feature_names)
# dump(rsf, '../../../Website/Models/rsf_model.joblib')
#
# print("*** 5 year Survival Predictor Demo ***")
# age_at_diagnosis_in_years = input("Please enter age: ")
# tnm_stage = input("Please enter TNM Stage (2/3): ")
# mmr_status = input("Please enter MMR Status (pMMR/dMMR): ")
# if mmr_status == "dMMR":
#     mmr_status = 1
# else:
#     mmr_status = 0
# cimp_status = input("Please enter CIMP Status (+/-): ")
# if cimp_status == "+":
#     cimp_status = 1
# else:
#     cimp_status = 0
# rfs_event_censored_5yr = input("Please enter whether relapse has occurred (y/n): ")
# if rfs_event_censored_5yr.lower() == "y":
#     rfs_event_censored_5yr = 1
# else:
#     rfs_event_censored_5yr = 0
# # if rfs_event_censored_5yr == 1:
# #     rfs_months_censored_5yr = input("Please enter how long after diagnosis relapse occurred in months: ")
# # else:
# #     rfs_months_censored_5yr = 0
# sex_Male = input("Please enter Sex (male/female): ")
# if sex_Male.lower() == "male":
#     sex_Male = True
# else:
#     sex_Male = False
# tumour_location_proximal = input("Please enter tumour location (proximal/distal): ")
# if tumour_location_proximal.lower() == "proximal":
#     tumour_location_proximal = True
# else:
#     tumour_location_proximal = False
# chemotherapy_adjuvant_Y = input("Please enter whether chemotherapy has been administered (y/n): ")
# if chemotherapy_adjuvant_Y.lower() == "y":
#     chemotherapy_adjuvant_Y = True
# else:
#     chemotherapy_adjuvant_Y = False
# kras_mutation_WT = input("Please enter KRAS Mutation Type (WT/M): ")
# if kras_mutation_WT.lower() == "wt":
#     kras_mutation_WT = True
# else:
#     kras_mutation_WT = False
#
# input_data = [float(age_at_diagnosis_in_years), tnm_stage,  mmr_status, cimp_status,  rfs_event_censored_5yr, sex_Male, tumour_location_proximal,  chemotherapy_adjuvant_Y, kras_mutation_WT]
#
# #float(rfs_months_censored_5yr),
# input_df = pd.DataFrame([input_data], columns=feature_names)
#
# survival_probabilities = rsf.predict_survival_function(input_df)
# time_points = survival_probabilities[0].x
# probabilities = survival_probabilities[0].y
# time_index1 = np.where(time_points == 12)[0][0]
# time_index2 = np.where(time_points == 36)[0][0]
# time_index3 = np.where(time_points == 60)[0][0]
# one_year_survival_probability = probabilities[time_index1]
# three_year_survival_probability = probabilities[time_index2]
# five_year_survival_probability = probabilities[time_index3]
# print(
#     f"1-year survival probability: {one_year_survival_probability}\n3-year survival probability: {three_year_survival_probability}\n5-year survival probability: {five_year_survival_probability}")
#
