import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Load data
features_file_paths = ([#'../../Files/10yr/FeatureSets/Best_Features_1.csv',
                        # '../../Files/10yr/FeatureSets/Best_Features_2.csv',
                        # '../../Files/10yr/FeatureSets/Best_Features_3.csv',
                        #'../../Files/10yr/Train_Preprocessed_Data.csv'
                        #'../../Files/10yr/FeatureSets/Best_Features_4.csv',
                        '../../Files/10yr/FeatureSets/Best_Features_5.csv',
                        #'../../Files/10yr/FeatureSets/Best_Features_6.csv',
                        #'../../Files/10yr/FeatureSets/Best_Features_7.csv',
                        # '../../Files/10yr/FeatureSets/Best_Features_8.csv',
                        # '../../Files/10yr/FeatureSets/Best_Features_9.csv',
                        #'../../Files/10yr/FeatureSets/Best_Features_10.csv',

    ])

scaler = StandardScaler()



feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)

    feature_dataframes.append(feature_dataframe)



# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)

feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
average_c_indices = []

for feature_sets in feature_dataframes:
    print(f"Feature set {feature_set_counter}: {(list(feature_sets.columns._data))}")
    c_indices = []
    fold_counter = 1
    average_c_index = 0
    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)
    for train_index, test_index in kf.split(features):
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        time_to_event_train, time_to_event_test = time_to_event_data[train_index], time_to_event_data[test_index]
        # Train the model
        rsf = RandomSurvivalForest(min_samples_split=2, n_estimators=400, min_samples_leaf=8,
                                   max_features= None, max_depth=3, random_state=40)
        rsf.fit(features_train, time_to_event_train)
        # Evaluate the model
        result = rsf.score(features_test, time_to_event_test)
        c_indices.append(result)
        print(f"Fold {fold_counter} Concordance Index: {result}")

        # Calculate Feature Importances
        importance_result = permutation_importance(rsf, features_test, time_to_event_test, n_repeats=15,
                                                   random_state=42)

        # Organize and print feature importances
        # importance_df = pd.DataFrame({
        #     'Feature': features_test.columns,
        #     'Importance': importance_result.importances_mean
        # }).sort_values(by='Importance', ascending=False)
        # print(importance_df)

        fold_counter = fold_counter + 1

    average_c_index = sum(c_indices) / len(c_indices)
    print(f"Feature set {feature_set_counter} Average Concordance Index: {average_c_index}\n")
    if average_c_index > best_feature_set_c_index:
        best_feature_set_c_index = average_c_index
        best_feature_set = feature_set_counter
    feature_set_counter = feature_set_counter + 1

print(f"\nBest Feature set: {best_feature_set}\nWith a average concordance index of: {best_feature_set_c_index}")

# Train Model on best features from cross validation
best_features_for_model = feature_dataframes[best_feature_set - 1]
features = best_features_for_model.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = best_features_for_model[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(
    index=False)

rsf = RandomSurvivalForest(min_samples_split=2, n_estimators=400, min_samples_leaf=8,
                                   max_features= None, max_depth=3, random_state=40)

rsf.fit(features, time_to_event_data)

# Load in test data
test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_10yr'] = test_data['os_event_censored_10yr'].astype(bool)

# Select best features from test data
best_feature_columns = best_features_for_model.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=
                                                                                                                  False)

# Run test data through the model
result = rsf.score(test_features, test_time_to_event_data)
print(f"Unseen test data Concordance Index: {result}")

# Ensure features of selected model are not too closely correlated
correlation_matrix = features.corr()
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features")
plt.show()
