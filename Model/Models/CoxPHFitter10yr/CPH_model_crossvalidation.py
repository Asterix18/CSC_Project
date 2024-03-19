import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sklearn.pipeline import make_pipeline


def plot_auc(t, a_scores, a_mean):
    # Plot AUC
    plt.plot(t, a_scores, "o-", label=f"RSF (mean AUC = {a_mean:.3f})")
    plt.xlabel("Months since diagnosis")
    plt.ylabel("time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.show()


def evaluate_model(t, x_test, y_test, y_train, model):
    # Concordance Index
    c_index = model.score(x_test, y_test)

    # Brier Score
    rsf_probs = np.row_stack([fn(t) for fn in model.predict_survival_function(x_test)])
    b_score = brier_score(y_train, y_test, rsf_probs, t)

    # AUC score
    risk_scores = model.predict(x_test)
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, t)

    return c_index, b_score[1], rsf_mean_auc, rsf_auc


# Load data
features_file_paths = (['../../Files/10yr/CPHFeatureSets/Best_Features_1.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_2.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_3.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_4.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_5.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_6.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_7.csv',
                        '../../Files/10yr/CPHFeatureSets/Best_Features_8.csv',
                        ])

feature_dataframes = []
for file_path in features_file_paths:
    feature_dataframe = pd.read_csv(file_path)
    feature_dataframe['os_event_censored_10yr'] = feature_dataframe['os_event_censored_10yr'].astype(bool)
    feature_dataframes.append(feature_dataframe)

# Define the number of folds for cross-validation
n_splits = 5

# Initialize KFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize Variables
times = (np.array([60, 119]))
feature_set_counter = 1
best_feature_set_c_index = 0
best_feature_set = None
feature_set_metrics = []

for feature_sets in feature_dataframes:
    print(f"\nFeature set {feature_set_counter}")
    c_index_scores = []
    b_scores = []
    auc_means = []
    aucs = []

    average_c_index = 0
    fold_counter = 1
    aggregated_feature_importances = None

    features = feature_sets.drop(['os_event_censored_10yr', 'os_months_censored_10yr'], axis=1)
    time_to_event_data = feature_sets[['os_event_censored_10yr', 'os_months_censored_10yr']].to_records(index=False)
    for train_index, test_index in skf.split(feature_sets, time_to_event_data['os_event_censored_10yr']):
        # Split data into training and testing sets for this fold
        x_training, x_validation = features.iloc[train_index], features.iloc[test_index]
        y_training, y_validation = time_to_event_data[train_index], time_to_event_data[test_index]

        # Create and fit the Cox Proportional Hazards model
        cph = make_pipeline(CoxPHSurvivalAnalysis())
        cph.fit(x_training, y_training)

        # Evaluate the model on the test set
        ci, bs, auc_mean, auc = evaluate_model(times, x_validation, y_validation, y_training, cph)
        c_index_scores.append(ci)
        b_scores.append(bs)
        auc_means.append(auc_mean)
        aucs.append(auc)

        # if aggregated_feature_importances is None:
        #     aggregated_feature_importances = cph.params_
        # else:
        #     aggregated_feature_importances += cph.params_
        print(f"fold {fold_counter}: c-i: {ci.round(5)}\t\t b-s: {np.mean(bs).round(5)}\t\t auc: {auc_mean.round(5)}")
        fold_counter = fold_counter + 1

    # Calculate the average Concordance index across all folds
    average_c_index = np.mean(c_index_scores)
    average_b_score = np.mean(b_scores)
    average_auc_mean = np.mean(auc_means)
    average_auc = sum(aucs) / len(aucs)

    df_current_feature_set = pd.DataFrame({
        'Feature Set': [feature_set_counter],
        '5-Fold C-Index': [average_c_index],
        '5-Fold B-Score': [average_b_score],
        '5-Fold AUC': [average_auc_mean]
    })
    feature_set_metrics.append(df_current_feature_set)

    # mean_feature_importances = aggregated_feature_importances / n_splits
    # print(f"Feature Importances: {mean_feature_importances}")

    if average_c_index > best_feature_set_c_index:
        best_feature_set_c_index = average_c_index
        best_feature_set = feature_set_counter

    plot_auc(times, average_auc, average_auc_mean)

    feature_set_counter = feature_set_counter + 1

df_summary = pd.concat(feature_set_metrics, ignore_index=True)

# Display the final table
print("\n\n", df_summary)

print(f"\nBest Feature set: {best_feature_set}\nWith a average concordance index of: {best_feature_set_c_index}")

