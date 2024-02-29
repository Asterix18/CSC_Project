import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

data = pd.read_csv('../../Files/5yr/FeatureSets/Best_Features_5.csv')
data['os_event_censored_5yr'] = data['os_event_censored_5yr'].astype(bool)
features = data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
time_to_event_data = data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)
rsf = RandomSurvivalForest(max_depth=3, max_features=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100, random_state=40)
rsf.fit(features, time_to_event_data)

# Load in test data
test_data = pd.read_csv('../../Files/5yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_5yr'] = test_data['os_event_censored_5yr'].astype(bool)

# Select best features from test data
best_feature_columns = data.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(
    index=False)

# Run test data through the model
result = rsf.score(test_features, test_time_to_event_data)
print(f"Test data Concordance Index: {result}")

# Assuming rsf is your trained RandomSurvivalForest model and test_features, test_time_to_event_data are your test data
result = permutation_importance(rsf, test_features, test_time_to_event_data, n_repeats=30, random_state=42)

# Organize and print feature importances
importance_df = pd.DataFrame({
    'Feature': test_features.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Display probabilities for first 5 and last 5 entries in test data (sorted by age)
X_test_sorted = test_features.sort_values(by=["age_at_diagnosis_in_years"])
X_test_sel = pd.concat((X_test_sorted.head(20), X_test_sorted.tail(20)))

survival = rsf.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(survival):
    plt.step(rsf.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.grid(True)
plt.title("Feature Set 5 Patient Survival Probabilities")
plt.show()
print(pd.Series(rsf.predict(X_test_sel)))

# Get the survival probability for an individual patient at 5 years
individual_test = X_test_sorted.tail(1)
survival_probabilities = rsf.predict_survival_function(individual_test)
time_points = survival_probabilities[0].x
probabilities = survival_probabilities[0].y
time_index = np.where(time_points == 60)[0][0]
five_year_survival_probability = probabilities[time_index]
print(f"5-year survival probability: {five_year_survival_probability}")

# Feature set 8 parameters = max_depth=None, max_features=None, min_samples_leaf=4, min_samples_split=5,
#                           n_estimators=200, random_state=40
# Feature set 9 parameters = max_depth=None, max_features=None, min_samples_leaf=8, min_samples_split=15,
#                           n_estimators=100, random_state=40
# Feature set 10 parameters = max_depth=5, max_features=None, min_samples_leaf=8, min_samples_split=10,
#                           n_estimators=100, random_state=40
# Feature set 11 parameters = max_depth=5, max_features=None, min_samples_leaf=8, min_samples_split=10,
#                           n_estimators=100, random_state=40
# Feature set 12 parameters = max_depth=5, max_features=None, min_samples_leaf=8, min_samples_split=2,
#                            n_estimators=100, random_state=40
