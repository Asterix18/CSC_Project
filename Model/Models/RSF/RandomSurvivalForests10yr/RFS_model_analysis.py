import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sklearn.model_selection import StratifiedKFold


best_parameters = {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 14, 'n_estimators': 500}


# Function to evaluate models
def evaluate_model(t, x_test, y_test, y_train, model):
    # Evaluate the model
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    rsf_probabilities = np.row_stack([fn(times) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, rsf_probabilities, t)

    # AUC score
    rsf_risk_scores = model.predict(x_test)
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf_risk_scores, t)

    return c_index, b_score[1], rsf_mean_auc, rsf_auc


# Read in data set
data = pd.read_csv('../../../Files/10yr/RSFFeatureSets/Feature_set_4_optimised.csv')
data['os_event_censored_10yr'] = data['os_event_censored_10yr'].astype(bool)
features = data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

# Set up data stores and variables
c_indices = []
brier_scores = []
auc_means_scores = []
auc_scores = []
times = np.array([60, 119])
fold_counter = 1

# K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
print(f"Fold\tC-Index\t\t\t\t\tBrier Score\t\t\t\tAUC")

# Conduct K fold cross validation with optimal parameters
for train_index, test_index in skf.split(features, time_to_event_data['os_event_censored_10yr']):
    features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
    time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

    # Initiate model with optimal parameters
    rsf_model_validate = RandomSurvivalForest(random_state=40, **best_parameters)

    # Fit Model
    rsf_model_validate.fit(features_train, time_to_event_train)

    # Train and Evaluate the model
    ci, bs, auc_mean, auc = evaluate_model(times, features_validation, time_to_event_validation, time_to_event_train,
                                           rsf_model_validate)
    c_indices.append(ci)
    brier_scores.append(bs)
    auc_means_scores.append(auc_mean)
    auc_scores.append(auc)

    print(f"{fold_counter}\t\t{ci}\t\t{np.mean(bs)}\t\t{auc_mean}")

    fold_counter = fold_counter + 1

# Calculate average metrics
average_c_index = np.mean(c_indices)
average_brier_scores = sum(brier_scores) / len(brier_scores)
average_brier_mean_score = np.mean(brier_scores)
average_auc_means_score = np.mean(auc_means_scores)
average_auc_scores = sum(auc_scores) / len(auc_scores)

# Create data frame for displaying metrics
df_validation_feature_set = pd.DataFrame({
    'Method': ["Cross Validation"],
    'C-Index': [average_c_index],
    'AUC 5yr': [average_auc_scores[0]],
    'AUC 10yr': [average_auc_scores[1]],
    'Brier 5yr': [average_brier_scores[0]],
    'Brier 10yr': [average_brier_scores[1]]
})

# *** Test final model against unseen data ***
# Load in test data
test_data = pd.read_csv('../../../Files/10yr/Test_Preprocessed_Data.csv')
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
rsf_model_test = RandomSurvivalForest(random_state=40, **best_parameters)
rsf_model_test.fit(features, time_to_event_data)

# Evaluate model against test data
ci_test, bs_test, auc_mean_test, auc_test = evaluate_model(times, test_features, test_time_to_event_data,
                                                           time_to_event_data, rsf_model_test)

# Create data frame for displaying metrics
df_test_feature_set = pd.DataFrame({
    'Method': ["Unseen Data"],
    'C-Index': ci_test,
    'AUC 5yr': [auc_test[0]],
    'AUC 10yr': [auc_test[1]],
    'Brier 5yr': [bs_test[0]],
    'Brier 10yr': [bs_test[1]]
})

metrics_tables = pd.concat([df_validation_feature_set, df_test_feature_set], ignore_index=True)

print("\n\nTable displaying metrics for cross validation and unseen data\n", metrics_tables)


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

print("\n\n\n*** Analysis Finished ***")

# dump(rsf_model_test, "../../../Website/Models/10yr_model.joblib", compress=3)

# Load in test data
individual_test = pd.read_csv('../../../Files/10yr/Individual_test.csv')

individual_test = pd.get_dummies(individual_test, drop_first=True)

individual_test = individual_test[best_feature_columns]

# Split labels and features
individual_test = individual_test.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)

print(individual_test)
survival_probabilities = rsf_model_test.predict_survival_function(individual_test)
time_points = survival_probabilities[0].x
probabilities = survival_probabilities[0].y
time_index = np.where(time_points == 60)[0][0]
five_year_survival_probability = probabilities[time_index]
print(f"5-year survival probability: {five_year_survival_probability}")