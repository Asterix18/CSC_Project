import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import set_config
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.svm import FastSurvivalSVM

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

estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=40)

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["os_event_censored_10yr"], y["os_event_censored_10yr"])


# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=40)
