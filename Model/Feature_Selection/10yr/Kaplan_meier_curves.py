from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# Load in data
data = pd.read_csv('../../Files/10yr/Train_Preprocessed_Data.csv')

# Create instance of kaplan meier fitter
kmf = KaplanMeierFitter()

#
confidence_intervals_off = False

x_axis_limits = (0, data['os_months_censored_10yr'].max())
y_axis_limits = (0, 1)

# Create datasets
# ****** All Data ******
df_basic = data[['os_months_censored_10yr', 'os_event_censored_10yr']]

# Fit the model
kmf.fit(durations=df_basic['os_months_censored_10yr'], event_observed=df_basic['os_event_censored_10yr'])
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - All Data')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/AllData.png")
plt.show()

# ****** Genders ******
# Males
df_males = data[data['sex'] == 'Male']
# Fit the model
kmf.fit(durations=df_males['os_months_censored_10yr'], event_observed=df_males['os_event_censored_10yr'], label="Male")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Females
df_females = data[data['sex'] == 'Female']
# Fit the model
kmf.fit(durations=df_females['os_months_censored_10yr'], event_observed=df_females['os_event_censored_10yr'],
        label="Female")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Gender')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.legend()
plt.grid(True)
plt.savefig("KaplanMeierCurves/Sex.png")
plt.show()

# ****** Ages ******
# Age <= 60
df_age_under_70 = data[data['age_at_diagnosis_in_years'] <= 70]
# Fit the model
kmf.fit(durations=df_age_under_70['os_months_censored_10yr'], event_observed=df_age_under_70['os_event_censored_10yr'],
        label="0 - 70")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Age > 60
df_age_over_70 = data[data['age_at_diagnosis_in_years'] > 70]
# Fit the model
kmf.fit(durations=df_age_over_70['os_months_censored_10yr'], event_observed=df_age_over_70['os_event_censored_10yr'],
        label="71 - 100")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Age')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/Age.png")
plt.show()

# ****** Chemotherapy ******
# Received Chemotherapy
received_chemo = data[data['chemotherapy_adjuvant'] == 'Y']
# Fit the model
kmf.fit(durations=received_chemo['os_months_censored_10yr'], event_observed=received_chemo['os_event_censored_10yr'],
        label="Chemotherapy")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Did not receive chemotherapy
did_not_receive_chemo = data[data['chemotherapy_adjuvant'] == 'N']
# Fit the model
kmf.fit(durations=did_not_receive_chemo['os_months_censored_10yr'],
        event_observed=did_not_receive_chemo['os_event_censored_10yr'],
        label="No chemotherapy")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Chemotherapy')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/Chemotherapy.png")
plt.show()

# ****** TP53 Mutation ******
# Wild-Type
wild_type = data[data['tp53_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=wild_type['os_months_censored_10yr'], event_observed=wild_type['os_event_censored_10yr'],
        label="Wild-type")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Mutated
mutated = data[data['tp53_mutation'] == 'M']
# Fit the model
kmf.fit(durations=mutated['os_months_censored_10yr'], event_observed=mutated['os_event_censored_10yr'],
        label="Mutated")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - TP53 Mutation')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/TP53.png")
plt.show()

# ****** TNM Stage ******
# Stage 2
stage_2_tnm = data[data['tnm_stage'] == 2]
# Fit the model
kmf.fit(durations=stage_2_tnm['os_months_censored_10yr'], event_observed=stage_2_tnm['os_event_censored_10yr'],
        label="Stage 2 TNM")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Stage 3
stage_3_tnm = data[data['tnm_stage'] == 3]
# Fit the model
kmf.fit(durations=stage_3_tnm['os_months_censored_10yr'], event_observed=stage_3_tnm['os_event_censored_10yr'],
        label="Stage 3 TNM")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - tnm stage')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/TNM_stage.png")
plt.show()

# ****** BRAF Mutation ******
# Wild-Type
braf_wild_type = data[data['braf_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=braf_wild_type['os_months_censored_10yr'], event_observed=braf_wild_type['os_event_censored_10yr'],
        label="Wild-type")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Mutated
braf_mutated = data[data['braf_mutation'] == 'M']
# Fit the model
kmf.fit(durations=braf_mutated['os_months_censored_10yr'], event_observed=braf_mutated['os_event_censored_10yr'],
        label="Mutated")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - BRAF Mutation')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/BRAF_Mutation.png")
plt.show()

# ****** KRAS Mutation ******
# Wild-Type
kras_wild_type = data[data['kras_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=kras_wild_type['os_months_censored_10yr'], event_observed=kras_wild_type['os_event_censored_10yr'],
        label="Wild-type")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Mutated
kras_mutated = data[data['kras_mutation'] == 'M']
# Fit the model
kmf.fit(durations=kras_mutated['os_months_censored_10yr'], event_observed=kras_mutated['os_event_censored_10yr'],
        label="Mutated")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - KRAS Mutation')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/KRAS_mutation.png")
plt.show()

# ****** Tumour Location ******
# Proximal
proximal = data[data['tumour_location'] == 'proximal']
# Fit the model
kmf.fit(durations=proximal['os_months_censored_10yr'], event_observed=proximal['os_event_censored_10yr'],
        label="Proximal")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Distal
distal = data[data['tumour_location'] == 'distal']
# Fit the model
kmf.fit(durations=distal['os_months_censored_10yr'], event_observed=distal['os_event_censored_10yr'],
        label="Distal")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Tumour Location')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/Tumour_location.png")
plt.show()

# ****** Relapse Event 5 year ******
# Relapsed
relapse = data[data['rfs_event_censored_5yr'] == 1]
# Fit the model
kmf.fit(durations=
        relapse['os_months_censored_10yr'], event_observed=relapse['os_event_censored_10yr'], label="Relapse")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Did not relapse
no_relapse = data[data['rfs_event_censored_5yr'] == 0]
# Fit the model
kmf.fit(durations=no_relapse['os_months_censored_10yr'], event_observed=no_relapse['os_event_censored_10yr'],
        label="No Relapse")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Relapse Event within 5 years')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/relapse5yr.png")
plt.show()

# ****** Relapse Event after 5 years ******
# Relapsed
relapse = data[data['rfs_event_censored_10yr'] == 1]
# Fit the model
kmf.fit(durations=
        relapse['os_months_censored_10yr'], event_observed=relapse['os_event_censored_10yr'], label="Relapse")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Did not relapse
no_relapse = data[data['rfs_event_censored_10yr'] == 0]
# Fit the model
kmf.fit(durations=no_relapse['os_months_censored_10yr'], event_observed=no_relapse['os_event_censored_10yr'],
        label="No Relapse")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Relapse Event after 5 years')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/relapse10yr.png")
plt.show()

# ****** MMR Status ******

dMMR = data[data['mmr_status'] == 1]
# Fit the model
kmf.fit(durations=
        dMMR['os_months_censored_10yr'], event_observed=dMMR['os_event_censored_10yr'], label="dMMR")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Distal
pMMR = data[data['mmr_status'] == 0]
# Fit the model
kmf.fit(durations=pMMR['os_months_censored_10yr'], event_observed=pMMR['os_event_censored_10yr'],
        label="pMMR")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - MMR Status')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/mmr_status.png")
plt.show()

# ****** CIMP Status ******
# Plus
cimp_plus = data[data['cimp_status'] == 1]
# Fit the model
kmf.fit(durations=
        cimp_plus['os_months_censored_10yr'], event_observed=cimp_plus['os_event_censored_10yr'], label="cimp_plus")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Minus
cimp_minus = data[data['cimp_status'] == 0]
# Fit the model
kmf.fit(durations=cimp_minus['os_months_censored_10yr'], event_observed=cimp_minus['os_event_censored_10yr'],
        label="cimp_minus")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - CIMP Status')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/cimp_status.png")
plt.show()

# ****** CIN Status ******
# Plus
cin_plus = data[data['cin_status'] == 1]
# Fit the model
kmf.fit(durations=
        cin_plus['os_months_censored_10yr'], event_observed=cin_plus['os_event_censored_10yr'], label="cin_plus")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Minus
cin_minus = data[data['cin_status'] == 0]
# Fit the model
kmf.fit(durations=cin_minus['os_months_censored_10yr'], event_observed=cin_minus['os_event_censored_10yr'],
        label="cin_minus")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - CIMP Status')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.savefig("KaplanMeierCurves/cin_status.png")
plt.show()

# ****** tnm.t ******
for tnm_t_category in data['tnm.t'].unique():
    mask = data['tnm.t'] == tnm_t_category
    kmf.fit(durations=data[mask]['os_months_censored_10yr'],
            event_observed=data[mask]['os_event_censored_10yr'],
            label=str(tnm_t_category))
    if confidence_intervals_off:
        kmf.plot_survival_function(ci_show=False)
    else:
        kmf.plot_survival_function()

plt.title('Survival curves by tnm.t category up to 5 years')
plt.xlabel('Time in months')
plt.ylabel('Survival Probability')
plt.legend(title='tnm.t category')
plt.savefig("KaplanMeierCurves/tnmt.png")
plt.show()

# ****** tnm.n ******
for tnm_n_category in data['tnm.n'].unique():
    mask = data['tnm.n'] == tnm_n_category
    kmf.fit(durations=data[mask]['os_months_censored_10yr'],
            event_observed=data[mask]['os_event_censored_10yr'],
            label=str(tnm_n_category))
    if confidence_intervals_off:
        kmf.plot_survival_function(ci_show=False)
    else:
        kmf.plot_survival_function()

plt.title('Survival curves by tnm.n category up to 5 years')
plt.xlabel('Time in months')
plt.ylabel('Survival Probability')
plt.legend(title='tnm.n category')
plt.savefig("KaplanMeierCurves/tnmn.png")
plt.show()


# ****** tnm.m ******
for tnm_n_category in data['tnm.m'].unique():
    mask = data['tnm.m'] == tnm_n_category
    kmf.fit(durations=data[mask]['os_months_censored_10yr'],
            event_observed=data[mask]['os_event_censored_10yr'],
            label=str(tnm_n_category))
    if confidence_intervals_off:
        kmf.plot_survival_function(ci_show=False)
    else:
        kmf.plot_survival_function()

plt.title('Survival curves by tnm.m category up to 5 years')
plt.xlabel('Time in months')
plt.ylabel('Survival Probability')
plt.legend(title='tnm.n category')
plt.savefig("KaplanMeierCurves/tnmm.png")
plt.show()


# ****** CMS ******
for CMS_category in data['CMS'].unique():
    mask = data['CMS'] == CMS_category
    kmf.fit(durations=data[mask]['os_months_censored_10yr'],
            event_observed=data[mask]['os_event_censored_10yr'],
            label=str(CMS_category))
    if confidence_intervals_off:
        kmf.plot_survival_function(ci_show=False)
    else:
        kmf.plot_survival_function()

plt.title('Survival curves by CMS category up to 5 years')
plt.xlabel('Time in months')
plt.ylabel('Survival Probability')
plt.legend(title='CMS')
plt.savefig("KaplanMeierCurves/CMS.png")
plt.show()

# ****** LogRank Tests ******
# LogRank test - Ages
results = logrank_test(df_age_under_70['os_months_censored_10yr'], df_age_over_70['os_months_censored_10yr'],
                       event_observed_A=df_age_under_70['os_event_censored_10yr'],
                       event_observed_B=df_age_over_70['os_event_censored_10yr'])
# Print the results
print('\n\nAge p-value:', results.p_value)

# LogRank test - Gender
results = logrank_test(df_males['os_months_censored_10yr'], df_females['os_months_censored_10yr'],
                       event_observed_A=df_males['os_event_censored_10yr'],
                       event_observed_B=df_females['os_event_censored_10yr'])
# Print the results
print('\n\nSex p-value:', results.p_value)

# LogRank test - Chemotherapy
results = logrank_test(received_chemo['os_months_censored_10yr'], did_not_receive_chemo['os_months_censored_10yr'],
                       event_observed_A=received_chemo['os_event_censored_10yr'],
                       event_observed_B=did_not_receive_chemo['os_event_censored_10yr'])
# Print the results
print('\n\nChemotherapy p-value:', results.p_value)

# LogRank test - TP53 Mutation
results = logrank_test(wild_type['os_months_censored_10yr'], mutated['os_months_censored_10yr'],
                       event_observed_A=wild_type['os_event_censored_10yr'],
                       event_observed_B=mutated['os_event_censored_10yr'])
# Print the results
print('\n\nTP53 Mutation p-value:', results.p_value)

# LogRank test - BRAF Mutation
results = logrank_test(braf_wild_type['os_months_censored_10yr'], braf_mutated['os_months_censored_10yr'],
                       event_observed_A=braf_wild_type['os_event_censored_10yr'],
                       event_observed_B=braf_mutated['os_event_censored_10yr'])
# Print the results
print('\n\nBRAF Mutation p-value:', results.p_value)

# LogRank test - KRAS Mutation
results = logrank_test(kras_wild_type['os_months_censored_10yr'], kras_mutated['os_months_censored_10yr'],
                       event_observed_A=kras_wild_type['os_event_censored_10yr'],
                       event_observed_B=kras_mutated['os_event_censored_10yr'])
# Print the results
print('\n\nKRAS Mutation p-value:', results.p_value)

# LogRank test - Tumour Location
results = logrank_test(proximal['os_months_censored_10yr'], distal['os_months_censored_10yr'],
                       event_observed_A=proximal['os_event_censored_10yr'],
                       event_observed_B=distal['os_event_censored_10yr'])
# Print the results
print('\n\nTumour Location p-value:', results.p_value)

# LogRank test - TNM Stage
results = logrank_test(stage_2_tnm['os_months_censored_10yr'], stage_3_tnm['os_months_censored_10yr'],
                       event_observed_A=stage_2_tnm['os_event_censored_10yr'],
                       event_observed_B=stage_3_tnm['os_event_censored_10yr'])
# Print the results
print('\n\nTNM stage p-value:', results.p_value)

# LogRank test - TNM Stage
results = logrank_test(relapse['os_months_censored_10yr'], no_relapse['os_months_censored_10yr'],
                       event_observed_A=relapse['os_event_censored_10yr'],
                       event_observed_B=no_relapse['os_event_censored_10yr'])
# Print the results
print('\n\nRelapse Event p-value:', results.p_value)

# LogRank test - MMR Status
results = logrank_test(pMMR['os_months_censored_10yr'], dMMR['os_months_censored_10yr'],
                       event_observed_A=pMMR['os_event_censored_10yr'],
                       event_observed_B=dMMR['os_event_censored_10yr'])
# Print the results
print('\n\nMMR Status p-value:', results.p_value)

# LogRank test - CIMP Status
results = logrank_test(cimp_plus['os_months_censored_10yr'], cimp_minus['os_months_censored_10yr'],
                       event_observed_A=cimp_plus['os_event_censored_10yr'],
                       event_observed_B=cimp_minus['os_event_censored_10yr'])
# Print the results
print('\n\nCIMP Status p-value:', results.p_value)

# LogRank test - CIN Status
results = logrank_test(cin_plus['os_months_censored_10yr'], cin_minus['os_months_censored_10yr'],
                       event_observed_A=cin_plus['os_event_censored_10yr'],
                       event_observed_B=cin_minus['os_event_censored_10yr'])
# Print the results
print('\n\nCIN Status p-value:', results.p_value)

# LogRank test - tnm.t
print("\n\n**** Pairwise Log-rank tests for tnm.t ****")

tnm_t_levels = data['tnm.t'].unique()
pairwise_combinations = combinations(tnm_t_levels, 2)
for level1, level2 in pairwise_combinations:
    group1 = data[data['tnm.t'] == level1]
    group2 = data[data['tnm.t'] == level2]
    result = logrank_test(group1['os_months_censored_10yr'], group2['os_months_censored_10yr'],
                          event_observed_A=group1['os_event_censored_10yr'],
                          event_observed_B=group2['os_event_censored_10yr'])
    print(f"Log-rank test between {level1} and {level2}: p = {result.p_value}")

# LogRank test - tnm.n
print("\n\n**** Pairwise Log-rank tests for tnm.n ****")

tnm_n_levels = data['tnm.n'].unique()
pairwise_combinations = combinations(tnm_n_levels, 2)
for level1, level2 in pairwise_combinations:
    group1 = data[data['tnm.n'] == level1]
    group2 = data[data['tnm.n'] == level2]
    result = logrank_test(group1['os_months_censored_10yr'], group2['os_months_censored_10yr'],
                          event_observed_A=group1['os_event_censored_10yr'],
                          event_observed_B=group2['os_event_censored_10yr'])
    print(f"Log-rank test between {level1} and {level2}: p = {result.p_value}")

# LogRank test - CMS
print("\n\n**** Pairwise Log-rank tests for CMS ****")

CMS_levels = data['CMS'].unique()
pairwise_combinations = combinations(CMS_levels, 2)
for level1, level2 in pairwise_combinations:
    group1 = data[data['CMS'] == level1]
    group2 = data[data['CMS'] == level2]
    result = logrank_test(group1['os_months_censored_10yr'], group2['os_months_censored_10yr'],
                          event_observed_A=group1['os_event_censored_10yr'],
                          event_observed_B=group2['os_event_censored_10yr'])
    print(f"Log-rank test between {level1} and {level2}: p = {result.p_value}")