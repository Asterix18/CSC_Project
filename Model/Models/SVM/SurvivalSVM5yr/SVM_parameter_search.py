import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM


# Function to split labels and features
def split_data(data_set):
    features = data_set.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
    time_to_event_data = data_set[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

    return features, time_to_event_data


# Custom scoring function for gridsearch
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["os_event_censored_5yr"], y["os_months_censored_5yr"], prediction)
    return result[0]


def find_best_alpha(data_set):
    x, y = split_data(data_set)
    # Set up grid search
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=40)
    estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=40)
    param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
    gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model, n_jobs=-1, refit=False, cv=cv)

    warnings.filterwarnings("ignore", category=UserWarning)
    gcv.fit(x, y)
    return gcv.best_params_


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

feature_set_counter = 1

for feature_sets in feature_dataframes:
    best_alpha = find_best_alpha(feature_sets)
    print(f"Feature set {feature_set_counter}:\n\tOptimal alpha: {find_best_alpha(feature_sets)}")
    feature_set_counter += 1
