import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, brier_score


# Function to evaluate model
def evaluate_model(t, x_test, y_test, y_train, model):
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    cph_probabilities = np.row_stack([fn(times) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, cph_probabilities, t)

    # AUC score
    risk_scores = model.predict(x_test)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, t)

    return c_index, b_score[1], cph_mean_auc, cph_auc



# Read in and set up data set
data = pd.read_csv('../../../Files/5yr/RSFFeatureSets/Best_Features_8.csv')
data['os_event_censored_5yr'] = data['os_event_censored_5yr'].astype(bool)
features = data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
time_to_event_data = data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)


# Set up data stores
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

# Calculate mean metrics
average_c_index = np.mean(c_indices)
average_brier_scores = sum(brier_scores) / len(brier_scores)
average_brier_mean_score = np.mean(average_brier_scores)
average_auc_means_score = np.mean(auc_means_scores)
average_auc_scores = sum(auc_scores) / len(auc_scores)

# Create metrics data frame
df_validation_feature_set = pd.DataFrame({
    'Method': ["Cross Validation"],
    'C-Index': [average_c_index],
    'Average B-Score': [average_brier_mean_score],
    'Average AUC': [average_auc_means_score],
})

print("\n\n\tTable for cross validated metrics\n", df_validation_feature_set)

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
cph = make_pipeline(CoxPHSurvivalAnalysis())
cph.fit(features, time_to_event_data)

cph2 = CoxPHSurvivalAnalysis()
cph2.fit(features, time_to_event_data)

# Check the proportional hazards assumption for all variables
cph_check_assumptions = CoxPHFitter()
cph_check_assumptions.fit(data, duration_col='os_months_censored_5yr', event_col='os_event_censored_5yr')
cph_check_assumptions.check_assumptions(data, p_value_threshold=0.05)

# Evaluate model against test data
ci_test, bs_test, auc_mean_test, auc_test = evaluate_model(times, test_features, test_time_to_event_data,
                                                           time_to_event_data, cph)

df_test_feature_set = pd.DataFrame({
    'Method': ["Unseen Data"],
    'C-Index': [ci_test],
    'Average B-Score': [sum(bs_test) / len(bs_test)],
    'Average AUC': [auc_mean_test]
})
metrics_tables = pd.concat([df_validation_feature_set, df_test_feature_set], ignore_index=True)

print("\n\n\tTable displaying metrics for crossvalidation and unseen data\n", metrics_tables)


# Further Analysis
# Display probabilities for first 5 and last 5 entries in test data (sorted by age)
X_test_sorted = test_features.sort_values(by=["age_at_diagnosis_in_years"])
X_test_sel = pd.concat((X_test_sorted.head(5), X_test_sorted.tail(5)))

survival = cph2.predict_survival_function(X_test_sorted, return_array=True)
for i, s in enumerate(survival):
    plt.step(cph2.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.grid(True)
plt.title("Patient Survival Probabilities")
plt.show()


plt.ylim(0, 1)
plt.show()

# # Load in test data
individual_test = pd.read_csv('../../../Files/5yr/Individual_test.csv')

individual_test = pd.get_dummies(individual_test, drop_first=True)

individual_test = individual_test[best_feature_columns]

# Remove Time to event data
individual_test = individual_test.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)

survival_probabilities = cph2.predict_survival_function(individual_test)
time_points = survival_probabilities[0].x
probabilities = survival_probabilities[0].y
time_index = np.where(time_points == 60)[0][0]
print(time_index)
five_year_survival_probability = probabilities[time_index]
print(f"\n5-year survival probability: {five_year_survival_probability}")
