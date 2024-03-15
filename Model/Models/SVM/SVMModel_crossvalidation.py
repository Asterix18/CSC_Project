import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold, StratifiedShuffleSplit
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["os_event_censored_10yr"], y["os_months_censored_10yr"], prediction)
    return result[0]

def find_best_alpha(data_set):
    x, y = split_data(data_set)
    # Set up grid search
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = FastSurvivalSVM(max_iter=1000, tol=None, random_state=40)
    param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
    gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)

    warnings.filterwarnings("ignore", category=UserWarning)
    gcv.fit(x, y)

    return gcv.best_params_

def cross_val_SVM(alpha, data_set):
    x, y = split_data(data_set)
    kf = KFold(n_splits=5, shuffle=True, random_state=40)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svmModel = FastSurvivalSVM(alpha=alpha, max_iter=1000, tol=None, random_state=40)
        svmModel.fit(x_train, y_train)


def split_data(data_set):
    features = data_set.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = data_set[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)
    return features, time_to_event_data


# Setup file paths
features_file_paths = (['../../Files/10yr/RSFFeatureSets/Best_Features_1.csv', #{'alpha': 0.000244140625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_2.csv', #{'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_3.csv', #{'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_4.csv', #{'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_5.csv', #{'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_6.csv', #{'alpha': 0.000244140625}
                        ])

# Load in data sets
feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)

kf = KFold(n_splits=5, shuffle=True,  random_state=40)
feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
average_c_indices = []
feature_set_metrics = []

for feature_sets in feature_dataframes:
    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)
    print(f"Feature set {feature_set_counter} optimal alpha: {find_best_alpha(feature_sets)}")
    feature_set_counter += 1









# for train_index, test_index in kf.split(features):
#         features_train, features_test = features.iloc[train_index], features.iloc[test_index]
#         time_to_event_train, time_to_event_test = time_to_event_data[train_index], time_to_event_data[test_index]
#
#         svmModel = FastSurvivalSVM(alpha=gcv.best_params_['alpha'], max_iter=1000, tol=None, random_state=40)
#         svmModel.fit(features_train, time_to_event_train)
#
#         c_index = score_survival_model(svmModel, features_test, time_to_event_test)
#         c_indices.append(c_index)
#
#     av_c_index = sum(c_indices)/len(c_indices)
#     df_current_feature_set = pd.DataFrame({
#         'Feature Set': [feature_set_counter],
#         'Alpha Score': [gcv.best_params_['alpha']],
#         'Average C-Index': [av_c_index]
#     })
#
#     feature_set_metrics.append(df_current_feature_set)
#     svmModel = FastSurvivalSVM(alpha=gcv.best_params_['alpha'], max_iter=1000, tol=None, random_state=40)
#     svmModel.fit(features, time_to_event_data)
#     feature_set_counter += 1
#
#
# # Concatenate all DataFrames in the list into a single DataFrame
# df_summary = pd.concat(feature_set_metrics, ignore_index=True)
#
# # Display the final table
# print(df_summary)

# def plot_performance(gcv):
#     n_splits = gcv.cv.n_splits
#     cv_scores = {"alpha": [], "test_score": [], "split": []}
#     order = []
#     for i, params in enumerate(gcv.cv_results_["params"]):
#         name = f'{params["alpha"]:.5f}'
#         order.append(name)
#         for j in range(n_splits):
#             vs = gcv.cv_results_[f"split{j}_test_score"][i]
#             cv_scores["alpha"].append(name)
#             cv_scores["test_score"].append(vs)
#             cv_scores["split"].append(j)
#     df = pd.DataFrame.from_dict(cv_scores)
#     _, ax = plt.subplots(figsize=(11, 6))
#     sns.boxplot(x="alpha", y="test_score", data=df, order=order, ax=ax)
#     _, xtext = plt.xticks()
#     for t in xtext:
#         t.set_rotation("vertical")
#     plt.show()
#
# plot_performance(gcv)


