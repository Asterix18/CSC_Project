from lifelines import CoxPHFitter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

prioritize_bestP = False

data = pd.read_csv('../../Files/10yr/Train_Preprocessed_Data.csv')
data = data.drop(['rfs_months_censored_10yr', 'tnm.m', 'tnm.n', 'tnm.t'], axis=1)
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables to dummy variables

significant_level = 0.1
all_features = set(data.columns) - {'os_months_censored_10yr', 'os_event_censored_10yr'}

best_features_set = {''}
best_p_value = float('inf')
for feature_to_exclude_initially in all_features:
    current_features = all_features - {feature_to_exclude_initially}

    while True:
        cph = CoxPHFitter()
        cph.fit(data[list(current_features) + ['os_months_censored_10yr', 'os_event_censored_10yr']],
                'os_months_censored_10yr', 'os_event_censored_10yr')
        p_values = cph.summary['p']
        max_p_value = p_values.max()

        if max_p_value < significant_level:
            if prioritize_bestP:
                if best_p_value > max_p_value:
                    best_p_value = max_p_value
                    best_features_set = current_features.copy()
            else:
                if len(best_features_set) < len(current_features):
                    best_p_value = max_p_value
                    best_features_set = current_features.copy()
                elif len(best_features_set) == len(current_features) and best_p_value > max_p_value:
                    best_p_value = max_p_value
                    best_features_set = current_features.copy()
            break
        else:
            # Remove the least significant feature
            least_significant = p_values.idxmax()
            current_features.remove(least_significant)

print("Best selected features:", best_features_set)
best_features_data = data[list(best_features_set) + ['os_months_censored_10yr', 'os_event_censored_10yr']]
cph.fit(best_features_data, duration_col='os_months_censored_10yr', event_col='os_event_censored_10yr')

# Display the summary of the Cox model
print(cph.summary)

# Check the proportional hazards assumption for all variables
cph.check_assumptions(best_features_data, p_value_threshold=0.05)

# Calculate the correlation matrix
correlation_matrix = best_features_data.drop(['os_months_censored_10yr', 'os_event_censored_10yr'],
                                             axis=1).corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features")
plt.show()

# Save the dataframe to a new CSV file
# best_features_data.to_csv('./Files/10yr/Best_Features_3.csv', index=False)
