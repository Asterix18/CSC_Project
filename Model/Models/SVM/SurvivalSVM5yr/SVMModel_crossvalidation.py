import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.svm import FastSurvivalSVM

# Load in test data
test_data = pd.read_csv('../../../Files/5yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_5yr'] = test_data['os_event_censored_5yr'].astype(bool)

# Split labels and features
unseen_features = test_data.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
unseen_time_to_event_data = test_data[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(
    index=False)


# Function to cross validate SVM model
def cross_val_svm(a, data_set):
    x, y = split_data(data_set)
    svm_model = FastSurvivalSVM(alpha=a, max_iter=1000, tol=None, random_state=40)
    svm_model.fit(x, y)
    best_feature_columns = x.columns
    unseen_data = unseen_features[best_feature_columns]
    c_index_unseen = svm_model.score(unseen_data, unseen_time_to_event_data)

    c_indices = []
    fold_counter = 1
    # Set up k fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    print(f"Fold\t\tC-Index")
    for train_index, test_index in skf.split(x, y['os_event_censored_5yr']):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm_model = FastSurvivalSVM(alpha=a, max_iter=1000, tol=None, random_state=40)
        svm_model.fit(x_train, y_train)
        c_index = svm_model.score(x_test, y_test)
        c_indices.append(c_index)

        #
        print(f"{fold_counter}\t\t\t{c_index}")
        fold_counter += 1
    return c_indices, c_index_unseen


# Function to split into features and labels
def split_data(data_set):
    features = data_set.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
    time_to_event_data = data_set[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

    return features, time_to_event_data


# Setup file paths
features_file_paths = (['../../../Files/5yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_6.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_7.csv',
                        '../../../Files/5yr/RSFFeatureSets/Best_Features_8.csv',
                        '../../../Files/5yr/RSFFeatureSets/Feature_set_5_Optimised.csv'
                        ])

# Load in data sets
feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_5yr'] = feature_dataframe['os_event_censored_5yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)

# Optimal alphas derived from hyper tuning
alphas = [0.000244140625, 0.0009765625, 0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625,
          0.000244140625, 0.000244140625]

feature_set_counter = 1
feature_set_metrics = []

# Calculate the concordance index for each set of features with optimal alphas applied
for feature_sets, alpha in zip(feature_dataframes, alphas):
    print(f"\n\nFeature set {feature_set_counter}:")
    ci, ci_unseen= cross_val_svm(alpha, feature_sets)
    print(f"Average C-Index: {np.mean(ci)}, C-index for unseen data: {ci_unseen}")
    feature_set_counter += 1


