import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold

best_parameters = {'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 14, 'n_estimators': 500,
                   "max_features": "sqrt", "bootstrap": True}


# Function to evaluate model
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


# Function to plot variance across folds
def plot_variance():
    plt.figure(figsize=(10, 6))
    folds = range(1,6)
    plt.plot(folds, c_indices, color='blue', label='C-Index')
    plt.plot(folds, brier_scores, color='red', label='Brier Score')
    plt.plot(folds, auc_means_scores, color='green', label='AUC')
    plt.title('Model Performance Metrics Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()

# Read in data set
data = pd.read_csv('../../../Files/5yr/RSFFeatureSets/Feature_set_5_Optimised.csv')
data['os_event_censored_5yr'] = data['os_event_censored_5yr'].astype(bool)
features = data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
time_to_event_data = data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

# Set up data stores and variables
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

    rsf_model_validate = RandomSurvivalForest(random_state=40, **best_parameters)

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

# Calculate mean metrics
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

# Plot the variance across folds
plot_variance()

# *** Test final model against unseen data ***
# Load in test data
test_data = pd.read_csv('../../../Files/5yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_5yr'] = test_data['os_event_censored_5yr'].astype(bool)

# Select best features from test data
best_feature_columns = data.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(
    index=False)

# Initiate model with optimal parameters
rsf_model_test = RandomSurvivalForest(random_state=40, **best_parameters)

rsf_model_test.fit(features, time_to_event_data)

# Evaluate model against test data
ci_test, bs_test, auc_mean_test, auc_test = evaluate_model(times, test_features, test_time_to_event_data,
                                                           time_to_event_data, rsf_model_test)

# Create data frame to display metrics
df_test_feature_set = pd.DataFrame({
    'Method': ["Unseen Data"],
    'C-Index': [ci_test],
    'Average B-Score': [np.mean(bs_test)],
    'Average AUC': [auc_mean_test],

})
metrics_tables = pd.concat([df_validation_feature_set, df_test_feature_set], ignore_index=True)

print("\n\n\tTable displaying metrics for cross validation and unseen data\n", metrics_tables)

# Further Analysis
# Display probabilities for all patients in the test set
survival = rsf_model_test.predict_survival_function(test_features, return_array=True)

for i, s in enumerate(survival):
    plt.step(rsf_model_test.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.grid(True)
plt.title("Feature Set 5 Patient Survival Probabilities")
plt.show()

# # Load in test data
individual_test = pd.read_csv('../../../Files/5yr/Individual_test.csv')

individual_test = pd.get_dummies(individual_test, drop_first=True)

individual_test = individual_test[best_feature_columns]

# Remove Time to event data
individual_test = individual_test.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)

survival_probabilities = rsf_model_test.predict_survival_function(individual_test)
time_points = survival_probabilities[0].x
probabilities = survival_probabilities[0].y
time_index = np.where(time_points == 60)[0][0]
five_year_survival_probability = probabilities[time_index]
print(f"\n5-year survival probability: {five_year_survival_probability}")
print(features.columns)

# dump(rsf_model_test, "../../../Website/Models/5yr_model.joblib", compress=3)
