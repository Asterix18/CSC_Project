import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sksurv.metrics import brier_score, cumulative_dynamic_auc


original = pd.read_csv('../../Files/5yr/Train_Preprocessed_Data.csv')
original = pd.get_dummies(original, drop_first=True)  # Convert categorical variables to dummy variables
# Set up file paths
features_file_paths = (['../../Files/5yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_6.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_7.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_8.csv',

                        ])
feature_dataframes = []

# Load in data sets
for file_path in features_file_paths:
    subset_df = pd.read_csv(file_path)
    subset_columns = subset_df.columns
    new_subset_df = original[subset_columns].copy()
    new_subset_df.to_csv(file_path, index=False)
