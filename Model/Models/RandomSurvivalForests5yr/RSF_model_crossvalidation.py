import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sksurv.metrics import brier_score, cumulative_dynamic_auc

# Set up file paths
features_file_paths = (['../../Files/5yr/RSFFeatureSets/Best_Features_1.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_2.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_3.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_4.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_5.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_6.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_7.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_8.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_9.csv',
                        '../../Files/5yr/RSFFeatureSets/Best_Features_10.csv',

                        ])
feature_dataframes = []

# Load in data sets
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_5yr'] = feature_dataframe['os_event_censored_5yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)


def evaluate_model(t, x_test, y_test, y_train, model):
    # Evaluate the model
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    rsf_probabilities = np.row_stack([fn(t) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, rsf_probabilities, t)

    # AUC score
    rsf_risk_scores = model.predict(x_test)
    try:
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf_risk_scores, t)
    except Exception as e:
        print("An error occurred:", e)
        rsf_mean_auc, rsf_auc = 0, 0

    return c_index, b_score[1], rsf_mean_auc, rsf_auc


def plot_auc(t, a_scores, a_mean):
    # Plot AUC
    plt.plot(t, a_scores, "o-", label=f"RSF (mean AUC = {a_mean:.3f})")
    plt.xlabel("Months since diagnosis")
    plt.ylabel("time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.show()


def get_importances(model, x_test, y_test):
    result = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=42)
    return result.importances_mean


# K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

# Initialise variables
feature_set_counter = 1
times = np.array([59])

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

    # Split data into features and time to event data
    features = feature_sets.drop(['os_event_censored_5yr', 'os_months_censored_5yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_5yr', 'os_months_censored_5yr']].to_records(index=False)

    print(f"Fold\tC-Index\t\t\t\t\tBrier Score\t\t\t\tAUC")

    # Conduct K fold cross validation
    for train_index, test_index in skf.split(features, time_to_event_data['os_event_censored_5yr']):
        features_train, features_validation = features.iloc[train_index], features.iloc[test_index]
        time_to_event_train, time_to_event_validation = time_to_event_data[train_index], time_to_event_data[test_index]

        # Train and Evaluate the model
        # Set base parameters
        rsf_validation = RandomSurvivalForest(min_samples_split=10, n_estimators=100, random_state=40)

        # Fit Model
        rsf_validation.fit(features_train, time_to_event_train)

        # Evaluate model
        ci, bs, auc_mean, auc = evaluate_model(times, features_validation, time_to_event_validation, time_to_event_train
                                               , rsf_validation)
        # Print fold metrics
        print(f"{fold_counter}\t\t{ci}\t\t{sum(bs) / len(bs)}\t\t{auc_mean}")

        c_indices.append(ci)
        brier_scores.append(bs)
        if auc_mean != 0:
            auc_means_scores.append(auc_mean)
            auc_scores.append(auc)

        # Get fold importances
        importances.append(get_importances(rsf_validation, features_validation, time_to_event_validation))

        fold_counter = fold_counter + 1

    # Calculate average metrics for feature set
    average_c_index = np.mean(c_indices)
    average_brier_scores = np.mean(brier_scores, axis=1)
    average_brier_mean_score = np.mean(average_brier_scores)
    average_auc_means_score = np.mean(auc_means_scores)
    average_auc_scores = sum(auc_scores) / len(auc_scores)

    df_current_feature_set = pd.DataFrame({
        'Feature Set': [feature_set_counter],
        '5-Fold C-Index': [average_c_index],
        '5-Fold B-Score': [average_brier_mean_score],
        '5-Fold AUC': [average_auc_means_score]
    })
    feature_set_metrics.append(df_current_feature_set)

    # Calculate average importances for feature set
    # avg_importances = sum(importances) / len(importances)
    # importance_df = pd.DataFrame({
    #     'Feature': features.columns,
    #     'Importance': avg_importances
    # }).sort_values(by='Importance', ascending=False)
    # print("\n", importance_df)

    # Plot AUC
    plot_auc(times, average_auc_scores, average_auc_means_score)

    feature_set_counter += 1

# Concatenate all DataFrames in the list into a single DataFrame
df_summary = pd.concat(feature_set_metrics, ignore_index=True)

# Display the final table
print("\n\n", df_summary)

print("\n\n\n*** Analysis Finished ***")
