from lifelines import CoxPHFitter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

prioritize_bestP = False

data = pd.read_csv('../../Files/5yr/Train_Preprocessed_Data.csv')

data = data.drop(['tnm.m', 'tnm_stage', 'CMS', 'rfs_event_censored_5yr'], axis=1)
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables to dummy variables

significant_level = 0.05
all_features = set(data.columns) - {'os_months_censored_5yr',
                                    'os_event_censored_5yr'}  # replace with your time and event column names

cph = CoxPHFitter()
cph.fit(data, duration_col='os_months_censored_5yr', event_col='os_event_censored_5yr')

# Display the summary of the Cox model
print(cph.summary)
# Check the proportional hazards assumption for all variables
cph.check_assumptions(data, p_value_threshold=0.05)

# Calculate the correlation matrix
correlation_matrix = data.drop(['os_months_censored_5yr', 'os_event_censored_5yr'],
                               axis=1).corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features")
plt.show()

# Save the dataframe to a new CSV file
# plt.savefig("../Files/5yr/CorrelationMatrixes/BestFeatures7.png")
# best_features_data.to_csv('../Files/5yr/Best_Features_7.csv', index=False)
