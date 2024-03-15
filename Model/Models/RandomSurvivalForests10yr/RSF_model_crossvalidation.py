import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc

# Setup file paths
features_file_paths = (['../../Files/10yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../Files/10yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../Files/10yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../Files/10yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../Files/10yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../Files/10yr/RSFFeatureSets/Best_Features_6.csv',
                        ])

# Load in data sets
feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)


def evaluate_model(x, y):
    # Evaluate the model

    # Set base parameters
    rsf = RandomSurvivalForest(min_samples_split=10, n_estimators=100, random_state=40)

    # Fit Model
    rsf.fit(features_train, time_to_event_train)

    # Concordance Index
    c_index = rsf.score(x, y)

    # Brier Score
    times = np.array([12, 60, 119])
    rsf_probs = np.row_stack([fn(times) for fn in rsf.predict_survival_function(features_validation)])
    brier_score = integrated_brier_score(time_to_event_data, y, rsf_probs, times)

    # AUC score
    rsf_chf_funcs = rsf.predict_cumulative_hazard_function(features_validation, return_array=False)
    rsf_risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(time_to_event_train, y, rsf_risk_scores, times)

    return c_index, brier_score, rsf_mean_auc, rsf_auc


# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)

feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
average_c_indices = []
feature_set_metrics = []

for feature_sets in feature_dataframes:
    print(f"\nFeature set {feature_set_counter}")
    c_indices = []
    brier_scores = []
    auc_means_scores = []
    auc_scores = []
    fold_counter = 1
    average_c_index = 0
    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

    # Conduct K fold cross validation
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

    plt.plot(np.array([12, 60, 119]), average_auc_scores, "o-", label=f"RSF (mean AUC = {average_auc_means_score:.3f})")
    plt.xlabel("Months since diagnosis")
    plt.ylabel("time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.show()

    df_current_feature_set = pd.DataFrame({
        'Feature Set': [feature_set_counter],
        '5-Fold C-Index': [average_c_index],
        '5-Fold B-Score': [average_brier_score],
        '5-Fold AUC': [average_auc_means_score]
    })

    feature_set_metrics.append(df_current_feature_set)
    feature_set_counter += 1

# Concatenate all DataFrames in the list into a single DataFrame
df_summary = pd.concat(feature_set_metrics, ignore_index=True)

# Display the final table
print("\n\n", df_summary)

# Train Model on best features from cross validation
# best_features_for_model = feature_dataframes[best_feature_set - 1]
# features = best_features_for_model.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
# time_to_event_data = best_features_for_model[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(
#     index=False)
#
# rsf = RandomSurvivalForest(min_samples_leaf=10, n_estimators=100, random_state=40)
#
# rsf.fit(features, time_to_event_data)
#
# # Load in test data
# test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
# test_data = pd.get_dummies(test_data, drop_first=True)
# test_data['os_event_censored_10yr'] = test_data['os_event_censored_10yr'].astype(bool)
#
# # Select best features from test data
# best_feature_columns = best_features_for_model.columns
# best_features_test_data = test_data[best_feature_columns]
#
# # Split labels and features
# test_features = best_features_test_data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
# test_time_to_event_data = (best_features_test_data[['os_event_censored_10yr', 'os_months_censored_10yr']].
#                            to_records(index=False))
#
# # Run test data through the model
# result = rsf.score(test_features, test_time_to_event_data)
#
# # times = np.array([12,60,119])
# #
# # print(times)
#
# rsf_probs = np.row_stack([fn(times) for fn in rsf.predict_survival_function(test_features)])
# score = integrated_brier_score(time_to_event_data, test_time_to_event_data, rsf_probs, times)
#
# rsf_chf_funcs = rsf.predict_cumulative_hazard_function(test_features, return_array=False)
# rsf_risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])
#
# rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(time_to_event_data, test_time_to_event_data, rsf_risk_scores, times)
#
# print(f"Unseen test data:\nConcordance Index: {result}\nBrier score: {score}\nAUC: {rsf_mean_auc}")
#
# plt.plot(times, rsf_auc, "o-", label=f"RSF (mean AUC = {rsf_mean_auc:.3f})")
# plt.xlabel("days from enrollment")
# plt.ylabel("time-dependent AUC")
# plt.legend(loc="lower center")
# plt.grid(True)
# plt.show()
