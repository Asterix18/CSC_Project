import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["os_event_censored_10yr"], y["os_months_censored_10yr"], prediction)
    return result[0]

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

# Load in test data
test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_10yr'] = test_data['os_event_censored_10yr'].astype(bool)

kf = KFold(n_splits=5, shuffle=True,  random_state=40)
feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
average_c_indices = []
feature_set_metrics = []

for feature_sets in feature_dataframes:
    c_indices = []

    #Set up grid search
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = FastSurvivalSVM(max_iter=1000, tol=None, random_state=40)
    param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
    gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)

    fold_counter = 1
    average_c_index = 0
    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

    # Select best features from test data
    best_feature_columns = feature_sets.columns
    best_features_test_data = test_data[best_feature_columns]

    # Split labels and features
    test_features = best_features_test_data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    test_time_to_event_data = best_features_test_data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(
        index=False)

    warnings.filterwarnings("ignore", category=UserWarning)
    gcv.fit(features, time_to_event_data)

    print(gcv.best_params_)
    svmModel = FastSurvivalSVM(alpha=gcv.best_params_['alpha'], max_iter=1000, tol=None, random_state=40)


    for train_index, test_index in kf.split(features):
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        time_to_event_train, time_to_event_test = time_to_event_data[train_index], time_to_event_data[test_index]

        svmModel = FastSurvivalSVM(alpha=gcv.best_params_['alpha'], max_iter=1000, tol=None, random_state=40)
        svmModel.fit(features_train, time_to_event_train)

        c_index = score_survival_model(svmModel, features_test, time_to_event_test)
        c_indices.append(c_index)

    av_c_index = sum(c_indices)/len(c_indices)
    print(f"Average concordance index for feature set {feature_set_counter}: {av_c_index}")
    svmModel = FastSurvivalSVM(alpha=gcv.best_params_['alpha'], max_iter=1000, tol=None, random_state=40)
    svmModel.fit(features, time_to_event_data)
    print(f"Feature set {feature_set_counter} unseen data concordance index: {score_survival_model(svmModel, test_features, test_time_to_event_data)}")
    feature_set_counter += 1



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


