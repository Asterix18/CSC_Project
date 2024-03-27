import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sklearn.model_selection import GridSearchCV


def split_data(data_set):
    features = data_set.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = data_set[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

    return features, time_to_event_data


def score_model(model, X, y):
    t = np.array([119])

    prediction = model.predict(X)

    c_index = model.score(X, y)

    # Brier Score
    rsf_probabilities = np.row_stack([fn(t) for fn in model.predict_survival_function(X)])
    b_score = brier_score(y, y, rsf_probabilities, t)

    score = c_index - (b_score[1][0] * 2)

    return score


def parameter_selection(dataset):
    feature_set, event_set = split_data(dataset)
    # Create RSF model instance
    rsf = RandomSurvivalForest(random_state=40)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5, n_jobs=-1, scoring=score_model)

    # Fit the grid search to the data
    grid_search.fit(feature_set, event_set)

    # Print the best parameters
    return grid_search.best_params_, grid_search.best_score_


# Set up grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 9, 15, None],
    'min_samples_split': [2, 6, 10, 14],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Setup file paths
features_file_paths = ([
    '../../Files/10yr/RSFFeatureSets/Best_Features_1.csv',
    '../../Files/10yr/RSFFeatureSets/Best_Features_2.csv',
    # Data set 2 best parameters: {'max_depth': 9, 'max_features': 'sqrt', 'min_samples_leaf': 1,
    # 'min_samples_split': 6, 'n_estimators': 100}
    '../../Files/10yr/RSFFeatureSets/Best_Features_3.csv',
    '../../Files/10yr/RSFFeatureSets/Best_Features_4.csv',
    # Data set 4 best parameters: {'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 4,
    # 'min_samples_split': 14, 'n_estimators': 300}
    '../../Files/10yr/RSFFeatureSets/Best_Features_5.csv',
    '../../Files/10yr/RSFFeatureSets/Best_Features_6.csv',
    '../../Files/10yr/RSFFeatureSets/Best_Features_7.csv',
    '../../Files/10yr/RSFFeatureSets/Best_Features_8.csv',
    # Data set 8 best parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1,
    # 'min_samples_split': 6, 'n_estimators': 300}
])

counter = 1
for file_path in features_file_paths:
    # Load in each data set
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    # Run grid search on chosen data sets
    best_parameters, best_score = parameter_selection(feature_dataframe)
    print(f"Data set {counter} best parameters: {best_parameters} and score: {best_score}")
    counter += 1
