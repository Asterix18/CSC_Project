import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.model_selection import KFold

def evaluate_model(x, y):
    # Evaluate the model

    # Set base parameters
    rsf_model = RandomSurvivalForest(max_depth=3, max_features=None, min_samples_leaf=8, min_samples_split=2, n_estimators=400,
                           random_state=40)

    # Fit Model
    rsf_model.fit(features_train, time_to_event_train)

    # Concordance Index
    c_index = rsf_model.score(x, y)

    # Brier Score
    times = np.array([12, 60, 119])
    rsf_probs = np.row_stack([fn(times) for fn in rsf_model.predict_survival_function(features_validation)])
    brier_score = integrated_brier_score(time_to_event_data, y, rsf_probs, times)

    # AUC score
    rsf_chf_funcs = rsf_model.predict_cumulative_hazard_function(features_validation, return_array=False)
    rsf_risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(time_to_event_train, y, rsf_risk_scores, times)

    return c_index, brier_score, rsf_mean_auc, rsf_auc

# Read in data set
data = pd.read_csv('../../Files/10yr/RSFFeatureSets/Best_Features_5.csv')
data['os_event_censored_10yr'] = data['os_event_censored_10yr'].astype(bool)
features = data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)


c_indices = []
brier_scores = []
auc_means_scores = []
auc_scores = []
fold_counter = 1

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)

for train_index, test_index in kf.split(features):
    features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
    time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

    # Train and Evaluate the model
    ci, bs, auc_mean, auc = evaluate_model(features_validation, time_to_event_validation)
    c_indices.append(ci)
    brier_scores.append(bs)
    auc_means_scores.append(auc_mean)
    auc_scores.append(auc)

    print(f"Fold: {fold_counter}\t\tC-Index: {ci}\t\tBrier Score:{bs}\t\tAUC: {auc_mean}")

    fold_counter = fold_counter + 1


average_c_index = sum(c_indices) / len(c_indices)
average_brier_score = sum(brier_scores) / len(brier_scores)
average_auc_means_score = sum(auc_means_scores) / len(auc_means_scores)
average_auc_scores = sum(auc_scores)/len(auc_scores)

df_current_feature_set = pd.DataFrame({
        '5-Fold C-Index': [average_c_index],
        '5-Fold B-Score': [average_brier_score],
        '5-Fold AUC': [average_auc_means_score]
})

plt.plot(np.array([12, 60, 119]), average_auc_scores, "o-", label=f"RSF (mean AUC = {average_auc_means_score:.3f})")
plt.xlabel("Months since diagnosis")
plt.ylabel("time-dependent AUC")
plt.legend(loc="lower center")
plt.grid(True)
plt.show()


# Load in test data
test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_10yr'] = test_data['os_event_censored_10yr'].astype(bool)

# Select best features from test data
best_feature_columns = data.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(
    index=False)

# # Run test data through the model
# result = rsf.score(test_features, test_time_to_event_data)
# print(f"Test data Concordance Index: {result}")
#
# # Assuming rsf is your trained RandomSurvivalForest model and test_features, test_time_to_event_data are your test data
# result = permutation_importance(rsf, test_features, test_time_to_event_data, n_repeats=30, random_state=42)
#
# # Organize and print feature importance's
# importance_df = pd.DataFrame({
#     'Feature': test_features.columns,
#     'Importance': result.importances_mean
# }).sort_values(by='Importance', ascending=False)
#
# print(importance_df)

# # Display probabilities for first 5 and last 5 entries in test data (sorted by age)
# X_test_sorted = test_features.sort_values(by=["age_at_diagnosis_in_years"])
# X_test_sel = pd.concat((X_test_sorted.head(5), X_test_sorted.tail(5)))

# survival = rsf.predict_survival_function(X_test_sel, return_array=True)
#
# for i, s in enumerate(survival):
#     plt.step(rsf.unique_times_, s, where="post", label=str(i))
# plt.ylabel("Survival probability")
# plt.xlabel("Time in months")
# plt.grid(True)
# plt.title("Feature Set 5 Patient Survival Probabilities")
# plt.show()
# print(pd.Series(rsf.predict(X_test_sel)))

# Get the survival probability for an individual patient at 5 years
# individual_test = X_test_sorted.head(1)
# survival_probabilities = rsf.predict_survival_function(individual_test)
# time_points = survival_probabilities[0].x
# probabilities = survival_probabilities[0].y
# time_index = np.where(time_points == 60)[0][0]
# five_year_survival_probability = probabilities[time_index]
# print(f"5-year survival probability: {five_year_survival_probability}")

# # Get the survival probability for an individual patient at 10 years
# time_index2 = np.where(time_points == 120)[0][0]
# ten_year_survival_probability = probabilities[time_index2]
# print(f"10-year survival probability: {ten_year_survival_probability}")
#
# # Run test data through the model
# result = rsf.score(test_features, test_time_to_event_data)
