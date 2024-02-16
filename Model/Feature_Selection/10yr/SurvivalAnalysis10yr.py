from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../../Files/10yr/Train_Preprocessed_Data.csv')

kmf = KaplanMeierFitter()
confidence_intervals_off = False

x_axis_limits = (0, data['os_months_censored_10yr'].max())
y_axis_limits = (0, 1)


#Create datasets
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
kmf.fit(durations=df_females['os_months_censored_10yr'], event_observed=df_females['os_event_censored_10yr'], label="Female")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Gender')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.legend();
plt.grid(True)
plt.show()


# ****** Ages ******
# Age <= 60
df_age_under_70 = data[data['age_at_diagnosis_in_years'] <= 70]
# Fit the model
kmf.fit(durations=df_age_under_70['os_months_censored_10yr'], event_observed=df_age_under_70['os_event_censored_10yr'], label="0 - 70")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Age > 60
df_age_over_70 = data[data['age_at_diagnosis_in_years'] > 70]
# Fit the model
kmf.fit(durations=df_age_over_70['os_months_censored_10yr'], event_observed=df_age_over_70['os_event_censored_10yr'], label="71 - 100")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Age')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.show()
print(len(df_age_under_70))
print(len(df_age_over_70))


# ****** Chemotherapy ******
# Chemo
received_chemo = data[data['chemotherapy_adjuvant'] == 'Y']
# Fit the model
kmf.fit(durations=received_chemo['os_months_censored_10yr'], event_observed=received_chemo['os_event_censored_10yr'], label="Chemotherapy")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# No Chemo
did_not_receive_chemo = data[data['chemotherapy_adjuvant'] == 'N']
# Fit the model
kmf.fit(durations=did_not_receive_chemo['os_months_censored_10yr'], event_observed=did_not_receive_chemo['os_event_censored_10yr'],
        label="No chemotherapy")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Chemotherapy')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.show()

# ****** TP53 Mutation ******
# Wild-Type
#data['tp53_mutation'] = data['tp53_mutation'].replace('N/A', 'WT')
wild_type = data[data['tp53_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=wild_type['os_months_censored_10yr'], event_observed=wild_type['os_event_censored_10yr'], label="Wild-type")
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
plt.show()

# ****** TNM Stage ******
# Stage 2
stage_2_tnm = data[data['tnm_stage'] == 2]
# Fit the model
kmf.fit(durations=stage_2_tnm['os_months_censored_10yr'], event_observed=stage_2_tnm['os_event_censored_10yr'], label="Stage 2 TNM")
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
plt.show()

# ****** BRAF Mutation ******
# Wild-Type
braf_wild_type = data[data['braf_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=braf_wild_type['os_months_censored_10yr'], event_observed=braf_wild_type['os_event_censored_10yr'], label="Wild-type")
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
plt.show()

# ****** KRAS Mutation ******
# Wild-Type
kras_wild_type = data[data['kras_mutation'] == 'WT']
# Fit the model
kmf.fit(durations=kras_wild_type['os_months_censored_10yr'], event_observed=kras_wild_type['os_event_censored_10yr'], label="Wild-type")
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
plt.show()

# ****** Tumour Location ******
# Proximal
proximal = data[data['tumour_location'] == 'proximal']
# Fit the model
kmf.fit(durations=proximal['os_months_censored_10yr'], event_observed=proximal['os_event_censored_10yr'], label="Proximal")
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
plt.show()

# ****** Relapse Event ******
# Relapsed
relapse = data[data['rfs_event'] == 1]
# Fit the model
kmf.fit(durations=
        relapse['os_months_censored_10yr'], event_observed=relapse['os_event_censored_10yr'], label="Relapse")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Did not relapse
no_relapse = data[data['rfs_event'] == 0]
# Fit the model
kmf.fit(durations=no_relapse['os_months_censored_10yr'], event_observed=no_relapse['os_event_censored_10yr'],
        label="No Relapse")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - Relapse Event')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
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
plt.show()

# ****** CIMP Status ******
#Plus
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
plt.show()

# ****** tnm.t ******
# Stage 2 and 3
tnm_t3 = data[data['tnm.t'] == 3]
# Fit the model
kmf.fit(durations=
        tnm_t3['os_months_censored_10yr'], event_observed=tnm_t3['os_event_censored_10yr'], label="T2 and T3")
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Stage 4
tnm_t4 = data[data['tnm.t'] == 4]
# Fit the model
kmf.fit(durations=tnm_t4['os_months_censored_10yr'], event_observed=tnm_t4['os_event_censored_10yr'],
        label="T4")
# Plot the survival curve
kmf.plot_survival_function()
plt.title('10 year Kaplan-Meier Survival Curve - tnm.t stage')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.xlim(x_axis_limits)
plt.ylim(y_axis_limits)
plt.grid(True)
plt.show()

# ****** LogRank Tests ******
# LogRank test - Ages
results = logrank_test(df_age_under_70['os_months_censored_10yr'], df_age_over_70['os_months_censored_10yr'],
                       event_observed_A=df_age_under_70['os_event_censored_10yr'],
                       event_observed_B=df_age_over_70['os_event_censored_10yr'])
# Print the results
print('Age p-value:', results.p_value)

# LogRank test - Gender
results = logrank_test(df_males['os_months_censored_10yr'], df_females['os_months_censored_10yr'],
                       event_observed_A=df_males['os_event_censored_10yr'],
                       event_observed_B=df_females['os_event_censored_10yr'])
# Print the results
print('Sex p-value:', results.p_value)

# LogRank test - Chemotherapy
results = logrank_test(received_chemo['os_months_censored_10yr'], did_not_receive_chemo['os_months_censored_10yr'],
                       event_observed_A=received_chemo['os_event_censored_10yr'],
                       event_observed_B=did_not_receive_chemo['os_event_censored_10yr'])
# Print the results
print('Chemotherapy p-value:', results.p_value)

# LogRank test - TP53 Mutation
results = logrank_test(wild_type['os_months_censored_10yr'], mutated['os_months_censored_10yr'],
                       event_observed_A=wild_type['os_event_censored_10yr'],
                       event_observed_B=mutated['os_event_censored_10yr'])
# Print the results
print('TP53 Mutation p-value:', results.p_value)

# LogRank test - BRAF Mutation
results = logrank_test(braf_wild_type['os_months_censored_10yr'], braf_mutated['os_months_censored_10yr'],
                       event_observed_A=braf_wild_type['os_event_censored_10yr'],
                       event_observed_B=braf_mutated['os_event_censored_10yr'])
# Print the results
print('BRAF Mutation p-value:', results.p_value)

# LogRank test - BRAF Mutation
results = logrank_test(kras_wild_type['os_months_censored_10yr'], kras_mutated['os_months_censored_10yr'],
                       event_observed_A=kras_wild_type['os_event_censored_10yr'],
                       event_observed_B=kras_mutated['os_event_censored_10yr'])
# Print the results
print('KRAS Mutation p-value:', results.p_value)

# LogRank test - Tumour Location
results = logrank_test(proximal['os_months_censored_10yr'], distal['os_months_censored_10yr'],
                       event_observed_A=proximal['os_event_censored_10yr'],
                       event_observed_B=distal['os_event_censored_10yr'])
# Print the results
print('Tumour Location p-value:', results.p_value)

# LogRank test - TNM Stage
results = logrank_test(stage_2_tnm['os_months_censored_10yr'], stage_3_tnm['os_months_censored_10yr'],
                       event_observed_A=stage_2_tnm['os_event_censored_10yr'],
                       event_observed_B=stage_3_tnm['os_event_censored_10yr'])
# Print the results
print('TNM stage p-value:', results.p_value)

# LogRank test - TNM Stage
results = logrank_test(relapse['os_months_censored_10yr'], no_relapse['os_months_censored_10yr'],
                       event_observed_A=relapse['os_event_censored_10yr'],
                       event_observed_B=no_relapse['os_event_censored_10yr'])
# Print the results
print('Relapse Event p-value:', results.p_value)

# LogRank test - MMR Status
results = logrank_test(pMMR['os_months_censored_10yr'], dMMR['os_months_censored_10yr'],
                       event_observed_A=pMMR['os_event_censored_10yr'],
                       event_observed_B=dMMR['os_event_censored_10yr'])
# Print the results
print('MMR Status p-value:', results.p_value)

# LogRank test - CIMP Status
results = logrank_test(cimp_plus['os_months_censored_10yr'], cimp_minus['os_months_censored_10yr'],
                       event_observed_A=cimp_plus['os_event_censored_10yr'],
                       event_observed_B=cimp_minus['os_event_censored_10yr'])
# Print the results
print('CIMP Status p-value:', results.p_value)

# LogRank test - CIN Status
results = logrank_test(cin_plus['os_months_censored_10yr'], cin_minus['os_months_censored_10yr'],
                       event_observed_A=cin_plus['os_event_censored_10yr'],
                       event_observed_B=cin_minus['os_event_censored_10yr'])
# Print the results
print('CIN Status p-value:', results.p_value)

# LogRank test - tnm.t
results = logrank_test(tnm_t3['os_months_censored_10yr'], tnm_t4['os_months_censored_10yr'],
                       event_observed_A=tnm_t3['os_event_censored_10yr'],
                       event_observed_B=tnm_t4['os_event_censored_10yr'])
# Print the results
print('tnm.t p-value:', results.p_value)