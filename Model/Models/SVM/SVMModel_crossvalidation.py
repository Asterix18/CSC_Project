import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sksurv.svm import FastSurvivalSVM
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

calc_alpha = False


def find_best_alpha(data_set):
    x, y = split_data(data_set)
    # Set up grid search
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = FastSurvivalSVM(max_iter=1000, tol=None, random_state=40)
    param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
    gcv = GridSearchCV(estimator, param_grid, scoring=FastSurvivalSVM.score, n_jobs=1, refit=False, cv=cv)

    warnings.filterwarnings("ignore", category=UserWarning)
    gcv.fit(x, y)
    return gcv.best_params_


# Function to cross validate SVM model
def cross_val_svm(a, data_set):
    x, y = split_data(data_set)
    c_indices = []
    fold_counter = 1
    # Set up k fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    print(f"Fold\t\tC-Index")
    for train_index, test_index in skf.split(x, y['os_event_censored_10yr']):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm_model = FastSurvivalSVM(alpha=a, max_iter=1000, tol=None, random_state=40)
        svm_model.fit(x_train, y_train)
        c_index = svm_model.score(x_test, y_test)
        c_indices.append(c_index)
        print(f"{fold_counter}\t\t\t{c_index}")
        fold_counter += 1
    return c_indices


def split_data(data_set):
    features = data_set.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = data_set[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

    return features, time_to_event_data


# Setup file paths
features_file_paths = (['../../Files/10yr/RSFFeatureSets/Best_Features_1.csv',  # {'alpha': 0.000244140625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_2.csv',  # {'alpha': 0.0009765625} ^
                        '../../Files/10yr/RSFFeatureSets/Best_Features_3.csv',  # {'alpha': 0.0009765625} ^
                        '../../Files/10yr/RSFFeatureSets/Best_Features_4.csv',  # {'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_5.csv',  # {'alpha': 0.0009765625}
                        '../../Files/10yr/RSFFeatureSets/Best_Features_6.csv',  # {'alpha': 0.000244140625}
                        ])

# Load in data sets
feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)

feature_set_counter = 1

if calc_alpha:
    for feature_sets in feature_dataframes:
        best_alpha = find_best_alpha(feature_sets)
        print(f"Feature set {feature_set_counter}:\n\tOptimal alpha: {find_best_alpha(feature_sets)}")
        feature_set_counter += 1

# Optimal alphas derived from the above for loop
alphas = [0.000244140625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.000244140625]

feature_set_counter = 1
feature_set_metrics = []

# Calculate the concordance index for each set of features with optimal alphas applied
for feature_sets, alpha in zip(feature_dataframes, alphas):
    print(f"\n\nFeature set {feature_set_counter}:")
    ci= cross_val_svm(alpha, feature_sets)
    print(f"Average C-Index: {np.mean(ci)}")
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
