import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2

data = pd.read_csv("../../Files/5yr/Train_Preprocessed_Data.csv")

data = data.drop(columns=['os_months_censored_5yr'])
data = pd.get_dummies(data, drop_first=True)

features = data.drop(columns=['os_event_censored_5yr'])
target_variable = data['os_event_censored_5yr']

k = 10

selector = SelectKBest(score_func=chi2, k=k)

features_selector = selector.fit_transform(features, target_variable)

selected_features = features.columns[selector.get_support(indices=True)]

# Get p-values for the selected features
p_values = selector.pvalues_

# Get the scores for the selected features
scores = selector.scores_

# Combine feature names, scores, and p-values
feature_scores = pd.DataFrame({'Feature': features.columns, 'Score': scores, 'p-value': p_values})

# Sort the dataframe by score
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Display the feature scores and p-values
print(feature_scores)

correlation_matrix = data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features")
plt.show()

category_counts = data['tnm.m'].value_counts()
new_labels = ['M' + str(int(category)) for category in category_counts.index]

plt.figure(figsize=(8, 6))
sns.barplot(x=new_labels, y=category_counts.values)
plt.title('Distribution of Categories in tnm.m Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

category_counts = data['tnm.t'].value_counts()
new_labels = ['T' + str(int(category)) for category in category_counts.index]

plt.figure(figsize=(8, 6))
sns.barplot(x=new_labels, y=category_counts.values)
plt.title('Distribution of Categories in tnm.t Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

category_counts = data['tnm.n'].value_counts()
new_labels = ['N' + str(int(category)) for category in category_counts.index]

plt.figure(figsize=(8, 6))
sns.barplot(x=new_labels, y=category_counts.values)
plt.title('Distribution of Categories in tnm.n Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

category_counts = data['tnm_stage'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Categories in stage Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
