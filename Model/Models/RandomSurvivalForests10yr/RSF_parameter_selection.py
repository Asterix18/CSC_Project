import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sklearn.model_selection import GridSearchCV

feature_dataframe = pd.read_csv('../../Files/10yr/RSFFeatureSets/Best_Features_8.csv')
feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
features = feature_dataframe.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = feature_dataframe[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)


def score_model(model, X, y):
    t = np.array([119])

    prediction = model.predict(X)

    c_index = model.score(X, y)

    # Brier Score
    rsf_probabilities = np.row_stack([fn(t) for fn in model.predict_survival_function(X)])
    b_score = brier_score(y, y, rsf_probabilities, t)

    _, rsf_mean_auc = cumulative_dynamic_auc(y, y, prediction, t)

    score = (rsf_mean_auc / 2) + c_index - (b_score[1][0] * 2)

    return score


param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 9, 15, None],
    'min_samples_split': [2, 6, 10, 14],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create RSF model instance
rsf = RandomSurvivalForest(random_state=40)

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5, n_jobs=-1, scoring=score_model)

# Fit the grid search to the data
grid_search.fit(features, time_to_event_data)

# Print the best parameters
print(f"Best Parameters: {grid_search.best_params_}")
