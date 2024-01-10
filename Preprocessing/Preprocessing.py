import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Read in patient dataframe
data = pd.read_csv('../Files/gse39582_n469_clinical_data.csv')

# Display how many missing values each column has
print(data.isnull().sum())

# Drop any columns where there are more than 100 missing values
data = data.dropna(axis=1, thresh=len(data) - 100)

# Drop the clinically insignificant columns
data = data.drop(data[['PDS_call', 'cit_molecular_subtype']], axis = 1)

# tnm.t setup
data['tnm.t'] = data['tnm.t'].replace('T1', 1)
data['tnm.t'] = data['tnm.t'].replace('T2', 2)
data['tnm.t'] = data['tnm.t'].replace('T3', 3)
data['tnm.t'] = data['tnm.t'].replace('T4', 4)

# tnm.n setup
data['tnm.n'] = data['tnm.n'].replace('N0', 0)
data['tnm.n'] = data['tnm.n'].replace('N1', 1)
data['tnm.n'] = data['tnm.n'].replace('N2', 2)
data['tnm.n'] = data['tnm.n'].replace('N3', 3)
data['tnm.n'] = data['tnm.n'].replace('N+', 4)

# tnm.m setup
data['tnm.m'] = data['tnm.m'].replace('M0', 0)
data['tnm.m'] = data['tnm.m'].replace('M1', 1)
data['tnm.m'] = data['tnm.m'].replace('MX', 2)

# Turn cin_status into a binary feature
data['cin_status'] = data['cin_status'].replace('-', 0)
data['cin_status'] = data['cin_status'].replace('+', 1)

# Turn cimp_status into a binary feature
data['cimp_status'] = data['cimp_status'].replace('-', 0)
data['cimp_status'] = data['cimp_status'].replace('+', 1)

# Turn mmr_status into a binary feature
data['mmr_status'] = data['mmr_status'].replace('pMMR', 0)
data['mmr_status'] = data['mmr_status'].replace('dMMR', 1)

# CMS setup
data['CMS'] = data['CMS'].replace('CMS1', 0)
data['CMS'] = data['CMS'].replace('CMS2', 1)
data['CMS'] = data['CMS'].replace('CMS3', 2)
data['CMS'] = data['CMS'].replace('CMS4', 3)
data['CMS'] = data['CMS'].replace('UNK', 4)

# Identify feature correlation to support filling in missing values
data_correlation = data.dropna()
data_correlation = pd.get_dummies(data_correlation, drop_first=True)
correlation_matrix = data_correlation.corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of all features")
#plt.show()

# Assign means/modes to missing values

# Fill in mmr_status missing values based off cimp_status (Correlation = 0.51)
mmr_status_for_cimp_0 = data[data['cimp_status'] == 0]['mmr_status'].mode()[0]
mmr_status_for_cimp_1 = data[data['cimp_status'] == 1]['mmr_status'].mode()[0]
data.loc[(data['cimp_status'] == 0) & (data['mmr_status'].isnull()), 'mmr_status'] = mmr_status_for_cimp_0
data.loc[(data['cimp_status'] == 1) & (data['mmr_status'].isnull()), 'mmr_status'] = mmr_status_for_cimp_1

#Fill in cimp_status missing values based off mmr_status
cimp_status_for_mmr_0 = data[data['mmr_status'] == 0]['cimp_status'].mode()[0]
cimp_status_for_mmr_1 = data[data['mmr_status'] == 1]['cimp_status'].mode()[0]
data.loc[(data['mmr_status'] == 0) & (data['cimp_status'].isnull()), 'cimp_status'] = cimp_status_for_mmr_0
data.loc[(data['mmr_status'] == 1) & (data['cimp_status'].isnull()), 'cimp_status'] = cimp_status_for_mmr_1

#Fill in cin_status missing values based off mmr_status (correlation = -0.48)
cin_status_for_mmr_0 = data[data['mmr_status'] == 0]['cin_status'].mode()[0]
cin_status_for_mmr_1 = data[data['mmr_status'] == 1]['cin_status'].mode()[0]
data.loc[(data['mmr_status'] == 0) & (data['cin_status'].isnull()), 'cin_status'] = cin_status_for_mmr_0
data.loc[(data['mmr_status'] == 1) & (data['cin_status'].isnull()), 'cin_status'] = cin_status_for_mmr_1

# Function to fill in tnm.t, tnm.n and tnm.m based off the modes of the values using tnm_stage a target
def tnm_fill_with_mode(df, tnm_stage, tnm_component):
    for stage in df[tnm_stage].unique():
        mode = df.loc[df[tnm_stage] == stage, tnm_component].mode()[0]
        df.loc[(df[tnm_stage] == stage) & (df[tnm_component].isna()), tnm_component] = mode
    return df

data = tnm_fill_with_mode(data, 'tnm_stage', 'tnm.t')
data = tnm_fill_with_mode(data, 'tnm_stage', 'tnm.n')
data = tnm_fill_with_mode(data, 'tnm_stage', 'tnm.m')

# Drop any empty chemotherapy_adjuvant rows (only 2)
data = data.dropna(subset=['chemotherapy_adjuvant'])

# Drop any rows where the patient has died or the study has ended after only 1 month
data = data[data['os_months'] > 1]

# Print summary and dataframe length to check there are no columns with missing values and the data frame is still a
# suitable size
print(data.isnull().sum())
print(len(data))

# Censoring os at 5 years (60 months) in order to calculate 5 year survival rates later
data['os_months_censored_5yr'] = data['os_months'].clip(upper=60)
data['os_event_censored_5yr'] = data.apply(lambda row: row['os_event'] if row['os_months'] <= 60 else 0, axis=1)

# Censoring os at 10 years (120 months) in order to calculate 10 year survival rates later
data['os_months_censored_10yr'] = data['os_months'].clip(upper=120)
data['os_event_censored_10yr'] = data.apply(lambda row: row['os_event'] if row['os_months'] <= 120 else 0, axis=1)

# Censoring rfs at 5 years (60 months) for calculating the 5 year survival rates
data['rfs_months_censored_5yr'] = data['rfs_months'].clip(upper=60)
data['rfs_event_censored_5yr'] = data.apply(lambda row: row['rfs_event'] if row['rfs_months'] <= 60 else 0, axis=1)

# Censoring rfs at 10 years (120 months) for calculating the 10 year survival rates
data['rfs_months_censored_10yr'] = data['rfs_months'].clip(upper=120)
data['rfs_event_censored_10yr'] = data.apply(lambda row: row['rfs_event'] if row['rfs_months'] <= 120 else 0, axis=1)

data.loc[data['rfs_event_censored_5yr'] == 0, 'rfs_months_censored_5yr'] = 0

# Split data into training and testing data prior to feature selection to ensure test data is unseen by the final model
train_data_5yr, test_data_5yr = train_test_split(data, test_size=0.15, random_state=40, stratify=data[
    'os_event_censored_5yr'])
train_data_10yr, test_data_10yr = train_test_split(data, test_size=0.15, random_state=40, stratify=data[
    'os_event_censored_10yr'])

train_data_5yr = train_data_5yr.drop(['os_months_censored_10yr', 'os_event_censored_10yr', 'rfs_months_censored_10yr',
                     'rfs_event_censored_10yr', 'rfs_event', 'rfs_months', 'os_event', 'os_months'], axis=1)
test_data_5yr = test_data_5yr.drop(['os_months_censored_10yr', 'os_event_censored_10yr', 'rfs_months_censored_10yr',
                    'rfs_event_censored_10yr', 'rfs_event', 'rfs_months', 'os_event', 'os_months'], axis=1)

train_data_10yr = train_data_10yr.drop(['os_months_censored_5yr', 'os_event_censored_5yr', 'rfs_months_censored_5yr',
                      'rfs_event_censored_5yr', 'rfs_event', 'rfs_months', 'os_event', 'os_months'], axis=1)
test_data_10yr = test_data_10yr.drop(['os_months_censored_5yr', 'os_event_censored_5yr', 'rfs_months_censored_5yr',
                    'rfs_event_censored_5yr', 'rfs_event', 'rfs_months', 'os_event', 'os_months'], axis=1)

# # Save the train and test sets into separate CSV files
train_data_5yr.to_csv('../Files/5yr/Train_Preprocessed_Data.csv', index=False)
test_data_5yr.to_csv('../Files/5yr/Test_Preprocessed_Data.csv', index=False)
train_data_10yr.to_csv('../Files/10yr/Train_Preprocessed_Data.csv', index=False)
test_data_10yr.to_csv('../Files/10yr/Test_Preprocessed_Data.csv', index=False)

train_count_1s = train_data_5yr['os_event_censored_5yr'].sum()
test_count_1s = test_data_5yr['os_event_censored_5yr'].sum()

counts = [train_count_1s, test_count_1s]
labels = ['Training Set', 'Testing Set']

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=counts)
plt.title('Distribution of death event across 5yr train and test sets')
plt.xlabel('Dataset')
plt.ylabel('Count of death events')
plt.show()

train_percentage = (train_count_1s/len(train_data_5yr['os_event_censored_5yr']))*100
test_percentage = (test_count_1s/len(test_data_5yr['os_event_censored_5yr']))*100


print(f"Train Death Event Percentage: {train_percentage}%\nTest Death Event Percentage: {test_percentage}%\n")

train_count_1s = train_data_10yr['os_event_censored_10yr'].sum()
test_count_1s = test_data_10yr['os_event_censored_10yr'].sum()

counts = [train_count_1s, test_count_1s]
labels = ['Training Set', 'Testing Set']

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=counts)
plt.title('Distribution of death event across 10yr train and test sets')
plt.xlabel('Dataset')
plt.ylabel('Count of death events')
plt.show()

train_percentage = (train_count_1s/len(train_data_10yr['os_event_censored_10yr']))*100
test_percentage = (test_count_1s/len(test_data_10yr['os_event_censored_10yr']))*100


print(f"Train Death Event Percentage: {train_percentage}%\nTest Death Event Percentage: {test_percentage}%\n")


