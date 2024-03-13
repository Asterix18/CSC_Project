from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
features_file_paths = (['../../Files/10yr/CPHFeatureSets/Best_Features_1.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_2.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_3.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_4.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_5.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_6.csv'
                        ])

feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframes.append(feature_dataframe)

# Define the number of folds for cross-validation
n_splits = 5

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None

for feature_sets in feature_dataframes:
    print(f"Feature set {feature_set_counter}: {(list(feature_sets.columns._data))}")
    c_index_scores = []
    average_c_index = 0
    fold_counter = 1
    aggregated_feature_importances = None

    for train_index, test_index in kf.split(feature_sets):
        # Split data into training and testing sets for this fold
        train_data, test_data = feature_sets.iloc[train_index], feature_sets.iloc[test_index]

        # Create and fit the Cox Proportional Hazards model
        cph = CoxPHFitter()
        cph.fit(train_data, duration_col='os_months_censored_10yr', event_col='os_event_censored_10yr')

        # Evaluate the model on the test set
        test_c_index = cph.score(test_data, scoring_method='concordance_index')
        c_index_scores.append(test_c_index)
        if aggregated_feature_importances is None:
            aggregated_feature_importances = cph.params_
        else:
            aggregated_feature_importances += cph.params_
        print(f"Concordance index for fold {fold_counter}: {test_c_index}")
        fold_counter = fold_counter + 1

    # Calculate the average Concordance index across all folds
    average_c_index = np.mean(c_index_scores)
    mean_feature_importances = aggregated_feature_importances / n_splits
    print(f"Feature set {feature_set_counter} Average Concordance Index: {average_c_index}\n")
    #print(f"Feature Importances: {mean_feature_importances}")
    if average_c_index > best_feature_set_c_index:
        best_feature_set_c_index = average_c_index
        best_feature_set = feature_set_counter
    feature_set_counter = feature_set_counter + 1


print(f"\nBest Feature set: {best_feature_set}\nWith a average concordance index of: {best_feature_set_c_index}")

# Train Model on best features from cross validation
best_features_for_model = feature_dataframes[best_feature_set - 1]

cph.fit(best_features_for_model, duration_col='os_months_censored_10yr', event_col='os_event_censored_10yr')

# Load in test data
test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)

# Select best features from test data
best_feature_columns = best_features_for_model.columns
best_features_test_data = test_data[best_feature_columns]

# Run test data through the model
result = cph.score(best_features_test_data, scoring_method='concordance_index')
print(f"Unseen test data Concordance Index: {result}")

# Check assumptions
cph.check_assumptions(best_features_for_model, p_value_threshold=0.05)

# Ensure features of selected model are not too closely correlated
best_features_for_model = best_features_for_model.drop(columns=['os_event_censored_10yr', 'os_months_censored_10yr'])
correlation_matrix = best_features_for_model.corr()
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features")
plt.show()
