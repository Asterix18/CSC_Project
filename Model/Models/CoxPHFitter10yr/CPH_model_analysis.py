import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, brier_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline


def evaluate_model(t, x_test, y_test, y_train, model):
    # Evaluate the model
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    cph_probs = np.row_stack([fn(times) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, cph_probs, t)

    # AUC score
    risk_scores = model.predict(x_test)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, t)

    return c_index, b_score[1], cph_mean_auc, cph_auc


def plot_auc(t, auc_scores, auc_mean):
    # Plot AUC
    plt.plot(t, auc_scores, "o-", label=f"RSF (mean AUC = {auc_mean:.3f})")
    plt.xlabel("Months since diagnosis")
    plt.ylabel("time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.show()


# Read in data set
data = pd.read_csv('../../Files/10yr/RSFFeatureSets/Best_Features_5.csv')
data['os_event_censored_10yr'] = data['os_event_censored_10yr'].astype(bool)
features = data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

c_indices = []
brier_scores = []
auc_means_scores = []
auc_scores = []
times = np.array([60, 119])
fold_counter = 1

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)
print(f"Fold\tC-Index\t\t\t\t\tBrier Score\t\t\t\tAUC")

# Conduct K fold cross validation with optimal parameters
for train_index, test_index in kf.split(features):
    features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
    time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

    rsf_model_validate = RandomSurvivalForest(max_depth=3, max_features=None, min_samples_leaf=8, min_samples_split=2,
                                              n_estimators=400, random_state=40)

    # Create and fit the Cox Proportional Hazards model
    cph = make_pipeline(CoxPHSurvivalAnalysis())
    cph.fit(features_train, time_to_event_train)

    # Train and Evaluate the model
    ci, bs, auc_mean, auc = evaluate_model(times, features_validation, time_to_event_validation, time_to_event_train,
                                           cph)
    c_indices.append(ci)
    brier_scores.append(bs)
    auc_means_scores.append(auc_mean)
    auc_scores.append(auc)

    print(f"{fold_counter}\t\t{ci}\t\t{sum(bs) / len(bs)}\t\t{auc_mean}")

    fold_counter = fold_counter + 1

average_c_index = sum(c_indices) / len(c_indices)
average_brier_scores = sum(brier_scores) / len(brier_scores)
average_brier_mean_score = sum(average_brier_scores) / len(average_brier_scores)
average_auc_means_score = sum(auc_means_scores) / len(auc_means_scores)
average_auc_scores = sum(auc_scores) / len(auc_scores)

df_validation_feature_set = pd.DataFrame({
    'Method': ["Cross Validation"],
    'C-Index': [average_c_index],
    'Average B-Score': [average_brier_mean_score],
    '5 year B-Score': [average_brier_scores[0]],
    '10 year B-Score': [average_brier_scores[1]],
    'Average AUC': [average_auc_means_score],
    '5 year AUC': [average_auc_scores[0]],
    '10 year AUC': [average_auc_scores[1]]
})

print("\n\n\tTable for cross validated metrics\n", df_validation_feature_set)

plot_auc(times, average_auc_scores, average_auc_means_score)

# *** Test final model against unseen data ***
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

# Initiate model with optimal parameters
cph = make_pipeline(CoxPHSurvivalAnalysis())
cph.fit(features, time_to_event_data)

# Evaluate model against test data
ci_test, bs_test, auc_mean_test, auc_test = evaluate_model(times, test_features, test_time_to_event_data,
                                                           time_to_event_data, cph)

df_test_feature_set = pd.DataFrame({
    'Method': ["Unseen Data"],
    'C-Index': [ci_test],
    'Average B-Score': [sum(bs_test) / len(bs_test)],
    '5 year B-Score': [bs_test[0]],
    '10 year B-Score': [bs_test[1]],
    'Average AUC': [auc_mean_test],
    '5 year AUC': [auc_test[0]],
    '10 year AUC': [auc_test[1]]
})
metrics_tables = pd.concat([df_validation_feature_set, df_test_feature_set], ignore_index=True)

print("\n\n\tTable displaying metrics for crossvalidation and unseen data\n", metrics_tables)

plot_auc(times, auc_test, auc_mean_test)

metrics_tables.to_csv("../../Files/tables and graphs/10yr_rsf_metrics.csv", index=False)


# Further Analysis
# Display probabilities for first 5 and last 5 entries in test data (sorted by age)
X_test_sorted = test_features.sort_values(by=["age_at_diagnosis_in_years"])
X_test_sel = pd.concat((X_test_sorted.head(5), X_test_sorted.tail(5)))

survival = cph.predict_survival_function(X_test_sel, return_array=True)
time_points = survival.time_points

for i, s in enumerate(survival):
    plt.step(time_points, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.grid(True)
plt.title("Feature Set 5 Patient Survival Probabilities")
plt.show()

# print(pd.Series(rsf_model_test.predict(X_test_sel)))

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

print("\n\n\n*** Analysis Finished ***")
