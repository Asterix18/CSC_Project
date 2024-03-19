import numpy as np
import pandas as pd
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score


def integrated_brier_score_survival_model(model, X, y, times):
    event_observed = y['os_event_censored_10yr']
    event_times = y['os_months_censored_10yr']
    predictions = model(X)

    score = integrated_brier_score(event_times, event_observed, predictions, times)

    return score
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["os_event_censored_10yr"], y["os_months_censored_10yr"], prediction)
    return result[0]

feature_dataframe = pd.read_csv('../../Files/10yr/RSFFeatureSets/Best_Features_4.csv')
feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)

features = feature_dataframe.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
time_to_event_data = feature_dataframe[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

# Load in test data
test_data = pd.read_csv('../../Files/10yr/Test_Preprocessed_Data.csv')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data['os_event_censored_10yr'] = test_data['os_event_censored_10yr'].astype(bool)

# Select best features from test data
best_feature_columns = feature_dataframe.columns
best_features_test_data = test_data[best_feature_columns]

# Split labels and features
test_features = best_features_test_data.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
test_time_to_event_data = best_features_test_data[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(
    index=False)

estimator = FastSurvivalSVM(alpha=0.0009765625, max_iter=1000, tol=None, random_state=40)
estimator.fit(features, time_to_event_data)
print(score_survival_model(estimator, test_features, test_time_to_event_data))
print(estimator.score(test_features, test_time_to_event_data))
times = np.array([12,60,119])
print(integrated_brier_score_survival_model(estimator, test_features, test_time_to_event_data, times))


