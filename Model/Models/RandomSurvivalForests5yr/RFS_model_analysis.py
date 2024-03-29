import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance


def evaluate_model(t, x_test, y_test, y_train, model):
    # Evaluate the model
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    rsf_probs = np.row_stack([fn(times) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, rsf_probs, t)

    rsf_risk_scores = model.predict(x_test)
    # AUC score
    try:
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf_risk_scores, t)
    except Exception as e:
        print("An error occurred:", e)
        rsf_mean_auc, rsf_auc = 0, 0

    return c_index, b_score[1], rsf_mean_auc, rsf_auc


def plot_auc(t, a_scores, a_mean):
    # Plot AUC
    plt.plot(t, a_scores, "o-", label=f"RSF (mean AUC = {a_mean:.3f})")
    plt.xlabel("Months since diagnosis")
    plt.ylabel("time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.show()


# Read in data set
data = pd.read_csv('../../Files/5yr/RSFFeatureSets/Best_Features_6.csv')
data['os_event_censored_5yr'] = data['os_event_censored_5yr'].astype(bool)
features = data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
time_to_event_data = data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

c_indices = []
brier_scores = []
auc_means_scores = []
auc_scores = []
times = np.array([59])
fold_counter = 1

# K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
print(f"Fold\tC-Index\t\t\t\t\tBrier Score\t\t\t\tAUC")

# Conduct K fold cross validation with optimal parameters
for train_index, test_index in skf.split(features, time_to_event_data['os_event_censored_5yr']):
    features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
    time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

    # Feature set 4
    # rsf_model_validate = RandomSurvivalForest(max_depth=9, min_samples_leaf=1, min_samples_split=14,
    #                                           n_estimators=300, random_state=40)

    # Feature set 6
    rsf_model_validate = RandomSurvivalForest(max_depth=3, min_samples_leaf=3, min_samples_split=2,
                                              n_estimators=300, random_state=40)

    # Fit Model
    rsf_model_validate.fit(features_train, time_to_event_train)

    # Train and Evaluate the model
    ci, bs, auc_mean, auc = evaluate_model(times, features_validation, time_to_event_validation, time_to_event_train,
                                           rsf_model_validate)
    c_indices.append(ci)
    brier_scores.append(bs)
    if auc_mean != 0:
        auc_means_scores.append(auc_mean)
        auc_scores.append(auc)

    print(f"{fold_counter}\t\t{ci}\t\t{bs[0]}\t\t{auc_mean}")

    fold_counter = fold_counter + 1

average_c_index = np.mean(c_indices)
average_brier_mean_score = np.mean(brier_scores)
average_auc_means_score = np.mean(auc_means_scores)
average_auc_scores = sum(auc_scores) / len(auc_scores)

df_validation_feature_set = pd.DataFrame({
    'Method': ["Cross Validation"],
    'C-Index': [average_c_index],
    'Average B-Score': [average_brier_mean_score],
    'Average AUC': [average_auc_means_score]
})

plot_auc(times, average_auc_scores, average_auc_means_score)

# *** Test final model against unseen data ***
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

# Initiate model with optimal parameters. Feature set 4
# rsf_model_test = RandomSurvivalForest(max_depth=9, min_samples_leaf=1, min_samples_split=14,
#                                               n_estimators=300, random_state=40)

# Initiate model with optimal parameters. Feature set 6
rsf_model_test = RandomSurvivalForest(max_depth=3, min_samples_leaf=3, min_samples_split=2,
                                              n_estimators=300, random_state=40)

rsf_model_test.fit(features, time_to_event_data)

# Evaluate model against test data
ci_test, bs_test, auc_mean_test, auc_test = evaluate_model(times, test_features, test_time_to_event_data,
                                                           time_to_event_data, rsf_model_test)

df_test_feature_set = pd.DataFrame({
    'Method': ["Unseen Data"],
    'C-Index': [ci_test],
    'Average B-Score': [np.mean(bs_test)],
    'Average AUC': [auc_mean_test],

})
metrics_tables = pd.concat([df_validation_feature_set, df_test_feature_set], ignore_index=True)

print("\n\n\tTable displaying metrics for cross validation and unseen data\n", metrics_tables)

plot_auc(times, auc_test, auc_mean_test)

metrics_tables.to_csv("../../Files/tables and graphs/5yr_rsf_metrics.csv", index=False)

# Further Analysis
# Display probabilities for first 5 and last 5 entries in test data (sorted by age)
X_test_sorted = test_features.sort_values(by=["age_at_diagnosis_in_years"])
X_test_sel = pd.concat((X_test_sorted.head(5), X_test_sorted.tail(5)))

survival = rsf_model_test.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(survival):
    plt.step(rsf_model_test.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.grid(True)
plt.title("Feature Set 5 Patient Survival Probabilities")
plt.show()

# print(pd.Series(rsf_model_test.predict(X_test_sel)))

# Load in test data
individual_test = pd.read_csv('../../Files/5yr/Individual_test.csv')

individual_test = pd.get_dummies(individual_test, drop_first=True)

individual_test = individual_test[best_feature_columns]

# Split labels and features
individual_test = individual_test.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)

print(individual_test)
survival_probabilities = rsf_model_test.predict_survival_function(individual_test)
time_points = survival_probabilities[0].x
probabilities = survival_probabilities[0].y
time_index = np.where(time_points == 60)[0][0]
five_year_survival_probability = probabilities[time_index]
print(f"5-year survival probability: {five_year_survival_probability}")

# dump(rsf_model_test, "../../../Website/Models/5yr_model.joblib", compress=3)
