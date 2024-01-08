import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV

features_file_paths = (['../../Files/5yr/FeatureSets/Best_Features_1.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_2.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_3.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_4.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_5.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_6.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_7.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_8.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_9.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_10.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_11.csv',
                        '../../Files/5yr/FeatureSets/Best_Features_12.csv'])

feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_5yr'] = feature_dataframe['os_event_censored_5yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)

data_set_num = 1

for dataset in feature_dataframes:
    features = dataset.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
    time_to_event_data = dataset[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

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
    print(f"Best Parameters for data set {data_set_num}: {grid_search.best_params_}")
    data_set_num = data_set_num + 1
