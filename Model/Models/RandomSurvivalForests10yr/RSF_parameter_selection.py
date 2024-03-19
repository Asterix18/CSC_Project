import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV

feature_dataframe = pd.read_csv('../../Files/10yr/RSFFeatureSets/Best_Features_8.csv')
feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
features = feature_dataframe.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = feature_dataframe[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)


param_grid = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Create RSF model instance
rsf = RandomSurvivalForest(random_state=40)

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(features, time_to_event_data)

# Print the best parameters
print(f"Best Parameters: {grid_search.best_params_}")
