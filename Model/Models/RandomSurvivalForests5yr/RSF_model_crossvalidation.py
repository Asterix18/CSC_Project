import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc

# Set up file paths
features_file_paths = (['../../Files/5yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_6.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_7.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_8.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_9.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_10.csv',

])
feature_dataframes = []

# Load in the data sets
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_5yr'] = feature_dataframe['os_event_censored_5yr'].astype(bool)
    if 'rfs_months_censored_5yr' in feature_dataframe.columns:
        feature_dataframe = feature_dataframe.drop(['rfs_months_censored_5yr'], axis=1)
    feature_dataframes.append(feature_dataframe)


# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)

feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
feature_set_metrics = []

for feature_sets in feature_dataframes:
    #print(f"Feature set {feature_set_counter}: {(list(feature_sets.columns._data))}")
    c_indices = []
    brier_scores = []
    fold_counter = 1
    average_c_index = 0
    features = feature_sets.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

    # Conduct K fold cross validation
    for train_index, test_index in kf.split(features):
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        time_to_event_train, time_to_event_test = time_to_event_data[train_index], time_to_event_data[test_index]
        # Train the model
        rsf = RandomSurvivalForest(min_samples_split=10, n_estimators=100, random_state=40)

        rsf.fit(features_train, time_to_event_train)

        # Evaluate the model
        # Concordance index
        result = rsf.score(features_test, time_to_event_test)
        c_indices.append(result)

        # Brier scores
        lower = 11
        upper = 60
        times = np.arange(lower, upper, 12)
        rsf_probs = np.row_stack([fn(times) for fn in rsf.predict_survival_function(features_test)])
        score = integrated_brier_score(time_to_event_data, time_to_event_test, rsf_probs, times)
        brier_scores.append(score)

        #print(f"Fold {fold_counter}: Concordance Index: {result.round(3)}, Brier Score: {score.round(3)}")

        # Calculate Feature Importances
        # importance_result = permutation_importance(rsf, features_test, time_to_event_test, n_repeats=15,
        #                                            random_state=42)
        # Organize and print feature importances
        # importance_df = pd.DataFrame({
        #     'Feature': features_test.columns,
        #     'Importance': importance_result.importances_mean
        # }).sort_values(by='Importance', ascending=False)
        # print(importance_df)

        fold_counter = fold_counter + 1

    # Calculate average metrics
    average_c_index = sum(c_indices)/len(c_indices)
    average_brier_score = sum(brier_scores)/len(brier_scores)

    df_current_feature_set = pd.DataFrame({
        'Feature Set': [feature_set_counter],
        'Average Concordance Index': [average_c_index],
        'Average Brier Score': [average_brier_score]
    })

    feature_set_metrics.append(df_current_feature_set)

    # print(f"Feature set {feature_set_counter}: Average Concordance Index: {average_c_index},"
    #       f" Average Brier Score: {average_brier_score}\n")

    if average_c_index > best_feature_set_c_index:
        best_feature_set_c_index = average_c_index
        best_feature_set = feature_set_counter
    feature_set_counter = feature_set_counter + 1

# Concatenate all DataFrames in the list into a single DataFrame
df_summary = pd.concat(feature_set_metrics, ignore_index=True)

# Display the final table
print(df_summary)

print(f"\nBest Feature set: {best_feature_set}\nWith a average concordance index of: {best_feature_set_c_index}")

# Train Model on best features from cross validation
best_features_for_model = feature_dataframes[best_feature_set - 1]
features = best_features_for_model.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
time_to_event_data = best_features_for_model[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(
    index=False)

rsf = RandomSurvivalForest(max_depth=3, max_features=None, min_samples_leaf=2, min_samples_split=15, n_estimators=100, random_state=40)

rsf.fit(features, time_to_event_data)

# Load in test data
test_data = pd.read_csv('../../Files/5yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_5yr'] = test_data['os_event_censored_5yr'].astype(bool)

# Select best features from test data
best_feature_columns = best_features_for_model.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=
                                                                                                                  False)

# Run test data through the model
result = rsf.score(test_features, test_time_to_event_data)
print(f"Unseen test data Concordance Index: {result}")


