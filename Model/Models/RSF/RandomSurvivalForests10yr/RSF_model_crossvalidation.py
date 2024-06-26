import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sksurv.metrics import cumulative_dynamic_auc, brier_score

# Load in test data
test_data = pd.read_csv('../../../Files/10yr/Individual_test.csv')

# Setup file paths
features_file_paths = (['../../../Files/10yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_6.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_7.csv',
                        '../../../Files/10yr/RSFFeatureSets/Best_Features_8.csv',
                        ])

# Load in data sets
feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)


# Function to evaluate models
def evaluate_model(t, x_test, y_test, y_train, model):
    # Evaluate the model
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    rsf_probabilities = np.row_stack([fn(t) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, rsf_probabilities, t)

    # AUC score
    rsf_risk_scores = model.predict(x_test)
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf_risk_scores, t)

    return c_index, b_score[1], rsf_mean_auc, rsf_auc


# Function to calculate feature importances
def get_importances(model, x_test, y_test):
    result = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=40)
    return result.importances_mean


# K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

# Initialise variables
feature_set_counter = 1
times = np.array([119])

# Initialise arrays
average_c_indices = []
feature_set_metrics = []

# Iterate through feature sets
for feature_sets in feature_dataframes:
    print(f"\nFeature set {feature_set_counter}")
    # Initialise arrays
    c_indices = []
    brier_scores = []
    auc_means_scores = []
    auc_scores = []
    importances = []

    # Initialise variables
    fold_counter = 1
    av_5yr_surv = 0

    # Split data into features and time to event data
    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)

    print(f"Fold\tC-Index\t\t\t\t\tBrier Score\t\t\t\tAUC")

    # Conduct K fold cross validation
    for train_index, test_index in skf.split(features, time_to_event_data['os_event_censored_10yr']):
        features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
        time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

        # Train and Evaluate the model
        # Set base parameters
        rsf_validation = RandomSurvivalForest(min_samples_split=10, n_estimators=100, random_state=40)

        # Fit Model
        rsf_validation.fit(features_train, time_to_event_train)

        # Evaluate model
        ci, bs, auc_mean, auc = evaluate_model(times, features_validation, time_to_event_validation,
                                               time_to_event_train, rsf_validation)
        # Print fold metrics
        print(f"{fold_counter}\t\t{ci}\t\t{bs[0]}\t\t{auc_mean}")

        c_indices.append(ci)
        brier_scores.append(bs)
        auc_means_scores.append(auc_mean)
        auc_scores.append(auc)

        # Get fold importances
        importances.append(get_importances(rsf_validation, features_validation, time_to_event_validation))

        # Check 5 year survival rates for individual patient to attempt to align 5 and 10 year models
        individual_test = test_data[features.columns]

        survival_probabilities = rsf_validation.predict_survival_function(individual_test)
        time_points = survival_probabilities[0].x
        probabilities = survival_probabilities[0].y
        time_index = np.where(time_points == 60)[0][0]
        five_year_survival_probability = probabilities[time_index]
        av_5yr_surv += five_year_survival_probability

        fold_counter = fold_counter + 1

    # Calculate average metrics for feature set
    av_5yr_surv = av_5yr_surv / 5
    average_c_index = np.mean(c_indices)
    average_brier_scores = np.mean(brier_scores, axis=1)
    average_brier_mean_score = np.mean(average_brier_scores)
    average_auc_means_score = np.mean(auc_means_scores)
    average_auc_scores = sum(auc_scores) / len(auc_scores)

    # Create data frame for displaying metrics
    df_current_feature_set = pd.DataFrame({
        'Feature Set': [feature_set_counter],
        '5-Fold C-Index': [average_c_index],
        '5-Fold B-Score': [average_brier_mean_score],
        '5-Fold AUC': [average_auc_means_score],
        '5 year Survival': av_5yr_surv
    })
    feature_set_metrics.append(df_current_feature_set)

    # Calculate average importances for feature set
    avg_importances = sum(importances) / len(importances)
    importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': avg_importances
    }).sort_values(by='Importance', ascending=False)
    print("\n", importance_df)

    feature_set_counter += 1

# Concatenate all DataFrames in the list into a single DataFrame
df_summary = pd.concat(feature_set_metrics, ignore_index=True)

# Display the final table
print("\n\n", df_summary)

print("\n\n\n*** Analysis Finished ***")
