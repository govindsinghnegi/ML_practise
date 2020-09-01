# import libraries here; add more as necessary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from IPython import display

# magic word for producing visualizations in notebook

import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# Load in the general demographics data.
azdias = pd.read_csv('data/Udacity_AZDIAS_Subset.csv', ';')
# Load in the feature summary file.
feat_info = pd.read_csv('data/AZDIAS_Feature_Summary.csv', ';')

# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print('azdias.shape: {}'.format(azdias.shape))
print('azdias.head: {}'.format(azdias.head(9).T))
print('azdias.describe: {}'.format(azdias.describe()))

# Identify missing or unknown data values and convert them to NaNs.
def missing_value_to_nan_converter(dataframe):
    for index, row in enumerate(dataframe.iteritems()):
        missing_or_unknown = feat_info['missing_or_unknown'][index]
        column_name = row[0]
        missing_or_unknown = missing_or_unknown[1:-1].split(',')
        if missing_or_unknown != ['']:
            missing_values = []
            for x in missing_or_unknown:
                if x in ['X', 'XX']:
                    missing_values.append(x)
                else:
                    missing_values.append(int(x))
        dataframe[column_name] = dataframe[column_name].replace(missing_values, np.nan)
    return dataframe

azdias = missing_value_to_nan_converter(azdias)
print('azdias.shape: {}'.format(azdias.shape))
print('azdias.head: {}'.format(azdias.head(9).T))
print('azdias.describe: {}'.format(azdias.describe()))


# Perform an assessment of how much missing data there is in each column of the
# dataset.
nan_per_col = azdias.isnull().sum()
nan_perc_per_col = (nan_per_col/azdias.shape[0]) * 100
azdias_nan_col = nan_perc_per_col.sort_values(ascending=False)
print('cols with percentage of null values : {}'.format(azdias_nan_col.head()))

# Investigate patterns in the amount of missing data in each column.
plt.figure(figsize=(20, 20))
azdias_nan_col.plot.bar();
plt.show()
azdias_no_nan_col = azdias_nan_col[azdias_nan_col == 0].index.values
print('cols having non-null values: {}'.format(azdias_no_nan_col))
for i in range(0, 30, 5):
    print('features with more than {} % missing values : {}'
          .format(i, len(azdias_nan_col[azdias_nan_col > i].index.values)))

# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
columns = azdias_nan_col[azdias_nan_col > 30].index.values
print('azdias_nan: {}'.format(columns))
azdias = azdias.drop(columns=columns)
print('azdias.shape: {}'.format(azdias.shape))
print('azdias.head: {}'.format(azdias.head(9).T))
print('azdias.describe: {}'.format(azdias.describe()))


# How much data is missing in each row of the dataset?
nan_per_row = azdias.isnull().sum(axis=1)
print('\n rows with number of null values: {}'.format(nan_per_row))
nan_perc_per_row = (nan_per_row/azdias.shape[1]) * 100
azdias_nan_row = nan_perc_per_row.sort_values(ascending=False)
print('\n rows with percentage of null values: {}'.format(azdias_nan_row))
azdias_no_nan_row = azdias_nan_row[azdias_nan_row == 0].index.values
print('\n rows having non-null values: {}'.format(azdias_no_nan_row))
print('\n percentage of rows having non-null values {} %'.format((len(azdias_no_nan_row)/azdias.shape[0])*100))
for i in range(0, 100, 10):
    print('number of rows with more than {} % missing values : {}'
          .format(i, len(azdias_nan_row[azdias_nan_row > i].index.values)))


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
missing_threshold = 30
azdias_above_threshold = azdias[azdias_nan_row > missing_threshold]
print('\n azdias_above_threshold.shape: {}'.format(azdias_above_threshold.shape))
print('\n azdias_above_threshold.head: \n{}'.format(azdias_above_threshold.head(9).T))
azdias_below_threshold = azdias[azdias_nan_row <= missing_threshold]
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold.head: \n{}'.format(azdias_below_threshold.head(9).T))


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
def subset_comparison(column):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    sns.countplot(x=column, data=azdias_above_threshold)
    plt.title('Above thresold')
    plt.subplot(122)
    sns.countplot(x=column, data=azdias_below_threshold)
    plt.title('Below thresold')
# randomly select 5 columns from columns which are not missing any data
random_no_nan_col = random.sample(list(azdias_no_nan_col), 5)
print('columns not missing any data: {}'.format(random_no_nan_col))
for random_col in random_no_nan_col:
    subset_comparison(random_col)

# How many features are there of each data type?
print('features type and count:\n {}'.format(feat_info['type'].value_counts()))
# first remove all deleted columns from feat_info dataset as well
for column in columns:
    feat_info = feat_info[feat_info.attribute != column]
print('after deletion new feat_info.shape: {}'.format(feat_info.shape))



# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
categorical_features = feat_info[feat_info.type == 'categorical'].attribute
print('categorical_features: \n{}'.format(categorical_features))
category_binary = []
category_multivar = []
for categorical_feature in categorical_features:
    if azdias_below_threshold[categorical_feature].nunique() <= 2:
        category_binary.append(categorical_feature)
    else:
        category_multivar.append(categorical_feature)
print('\nbinary categorical features: \n{}'.format(category_binary))
print('\nmultivar categorical features: \n{}'.format(category_multivar))


# Re-encode categorical variable(s) to be kept in the analysis.
for column in category_binary:
    print('\n {}'.format(azdias_below_threshold[column].value_counts()))

azdias_below_threshold['OST_WEST_KZ'] = azdias_below_threshold['OST_WEST_KZ'].map({'W': 0, 'O': 1})
print(azdias_below_threshold['OST_WEST_KZ'].value_counts())
for column in category_multivar:
    print('\n {}'.format(azdias_below_threshold[column].value_counts()))

azdias_below_threshold = azdias_below_threshold.drop(['CAMEO_DEU_2015'], axis=1)
del category_multivar[-1]
azdias_below_threshold = pd.get_dummies(azdias_below_threshold, columns = category_multivar)
print('\nafter encoding azdias_below_threshold.shape: {}\n'.format(azdias_below_threshold.shape))
print(azdias_below_threshold.head(9).T)


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
print('mixed_features: \n{}'.format(feat_info[feat_info.type == 'mixed'].attribute))
print('\n PRAEGENDE_JUGENDJAHRE: \n{}'.format(azdias_below_threshold['PRAEGENDE_JUGENDJAHRE'].value_counts()))
movement = []
mainstream = [1, 3, 5, 8, 10, 12, 14]
avantgrade = [2, 4, 6, 7, 9, 11, 13, 15]
for row in azdias_below_threshold['PRAEGENDE_JUGENDJAHRE']:
    if row in mainstream:
        movement.append(0)
    elif row in avantgrade:
        movement.append(1)
    else:
        movement.append(np.nan)
azdias_below_threshold['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = np.array(movement)
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold.head: \n{}'.format(azdias_below_threshold.head(9).T))



generation_mapping = {1: 1, 2: 1,
                      3: 2, 4: 2,
                      5: 3, 6: 3,
                      7: 3,
                      8: 4, 9: 4,
                      10: 5, 11: 5, 12: 5, 13: 5,
                      14: 6, 15: 6}
azdias_below_threshold['PRAEGENDE_JUGENDJAHRE_GENERATION'] = azdias_below_threshold['PRAEGENDE_JUGENDJAHRE'].map(generation_mapping)
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold[PRAEGENDE_JUGENDJAHRE_GENERATION].value_counts(): {}'
      .format(azdias_below_threshold['PRAEGENDE_JUGENDJAHRE_GENERATION'].value_counts()))

azdias_below_threshold = azdias_below_threshold.drop('PRAEGENDE_JUGENDJAHRE', axis=1)

# Investigate "CAMEO_INTL_2015" and engineer two new variables.
print('\n azdias_below_threshold[CAMEO_INTL_2015].value_counts(): {}'
      .format(azdias_below_threshold['CAMEO_INTL_2015'].value_counts()))
cameo_wealth = []
for x in azdias_below_threshold['CAMEO_INTL_2015']:
    if pd.isnull(x):
        cameo_wealth.append(np.nan)
        continue
    x = int(x)
    if 11 <= x <= 15:
        cameo_wealth.append(1)
    elif 21 <= x <= 25:
        cameo_wealth.append(2)
    elif 31 <= x <= 35:
        cameo_wealth.append(3)
    elif 41 <= x <= 45:
        cameo_wealth.append(4)
    elif 51 <= x <= 55:
        cameo_wealth.append(5)
    else:
        cameo_wealth.append(np.nan)
azdias_below_threshold['CAMEO_INTL_2015_WEALTH'] = np.array(cameo_wealth)
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold[CAMEO_INTL_2015_WEALTH].value_counts(): {}'.
      format(azdias_below_threshold['CAMEO_INTL_2015_WEALTH'].value_counts()))


cameo_lifestage = []
for x in azdias_below_threshold['CAMEO_INTL_2015']:
    if pd.isnull(x):
        cameo_lifestage.append(np.nan)
        continue
    x = int(x)
    if x % 10 == 1:
        cameo_lifestage.append(1)
    elif x % 10 == 2:
        cameo_lifestage.append(2)
    elif x % 10 == 3:
        cameo_lifestage.append(3)
    elif x % 10 == 4:
        cameo_lifestage.append(4)
    elif x % 10 == 5:
        cameo_lifestage.append(5)
    else:
        cameo_lifestage.append(np.nan)
azdias_below_threshold['CAMEO_INTL_2015_LIFESTAGE'] = np.array(cameo_lifestage)
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold[CAMEO_INTL_2015_LIFESTAGE].value_counts(): {}'.
      format(azdias_below_threshold['CAMEO_INTL_2015_LIFESTAGE'].value_counts()))


print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold.head: \n{}'.format(azdias_below_threshold.head(8).T))
azdias_below_threshold = azdias_below_threshold.drop('CAMEO_INTL_2015', axis=1)
print('\n azdias_below_threshold.shape: {}'.format(azdias_below_threshold.shape))
print('\n azdias_below_threshold.head: \n{}'.format(azdias_below_threshold.head(8).T))


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
azdias_below_threshold_copy = azdias_below_threshold.copy()
nan_per_col = azdias_below_threshold_copy.isnull().sum()
nan_perc_per_col = (nan_per_col/azdias_below_threshold_copy.shape[0]) * 100
azdias_nan_col = nan_perc_per_col.sort_values(ascending=False)
print('cols with percentage of null values : \n{}'.format(azdias_nan_col.head(50)))
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
azdias_imputed = imputer.fit_transform(azdias_below_threshold_copy)
azdias_imputed = pd.DataFrame(azdias_imputed, columns=azdias_below_threshold.columns)
print('\n after imputaton \n')
nan_per_col = azdias_imputed.isnull().sum()
nan_perc_per_col = (nan_per_col/azdias_imputed.shape[0]) * 100
azdias_nan_col = nan_perc_per_col.sort_values(ascending=False)
print('cols with percentage of null values : \n{}'.format(azdias_nan_col.head(50)))



print('azdias_imputed.shape: {}'.format(azdias_imputed.shape))
# Apply feature scaling to the general population demographics data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
azdias_scaled = pd.DataFrame(scaler.fit_transform(azdias_imputed), columns=azdias_below_threshold.columns)
print('\n azdias_scaled.shape: {}'.format(azdias_scaled.shape))
print('\n azdias_scaled.head: \n{}'.format(azdias_scaled.head(4).T))
print('\n azdias_scaled.describe: \n{}'.format(azdias_scaled.describe()))


# Apply PCA to the data.
pca = PCA()
azdias_pca = pca.fit_transform(azdias_scaled)


# Investigate the variance accounted for by each principal component.
def screen_plot(pca, show_labels=False):
    '''
    Creates a scree plot associated with the principal components
    INPUT: pca - the result of instantian of PCA in scikit learn
    OUTPUT: None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    if show_labels:
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


screen_plot(pca)




for i in range(1, 90, 10):
    print('\n {} principal components explains {} % of variance'
          .format(i, pca.explained_variance_ratio_[:i].sum()))


# Re-apply PCA to the data while selecting for number of components to retain.
pca = PCA(n_components=70)
azdias_pca = pca.fit_transform(azdias_scaled)
screen_plot(pca)



print('pca.components_.shape: {}'.format(pca.components_.shape))
print('\n pca.components_ : \n {}'.format(pca.components_))
print('\n pca.explained_variance_ratio_ : \n{}'.format(pca.explained_variance_ratio_))
# https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
component_to_feature_df = pd.DataFrame(pca.components_, columns=azdias_scaled.columns)
print('\n component_to_feature_df.head() : \n {}'.format(component_to_feature_df.head(4).T))



def map_feature_to_component(df, features, principal_component):
    print('Highest positive features: \n{}'.format(df.iloc[principal_component-1].sort_values(ascending=False)[:features]))
    print('\n Highest negative features: \n{}'.format(df.iloc[principal_component-1].sort_values(ascending=True)[:features]))



map_feature_to_component(component_to_feature_df, 5, 1)
map_feature_to_component(component_to_feature_df, 5, 2)
map_feature_to_component(component_to_feature_df, 5, 3)

###############################################clean_data()#########################


feat_info = pd.read_csv('data/AZDIAS_Feature_Summary.csv', ';')

def missing_value_to_nan_converter(dataframe):
    for index, row in enumerate(dataframe.iteritems()):
        missing_or_unknown = feat_info['missing_or_unknown'][index]
        column_name = row[0]
        missing_or_unknown = missing_or_unknown[1:-1].split(',')
        if missing_or_unknown != ['']:
            missing_values = []
            for x in missing_or_unknown:
                if x in ['X', 'XX']:
                    missing_values.append(x)
                else:
                    missing_values.append(int(x))
        dataframe[column_name] = dataframe[column_name].replace(missing_values, np.nan)

columns = []

azdias = azdias.drop(columns=columns)

nan_per_row = azdias.isnull().sum(axis=1)
nan_perc_per_row = (nan_per_row/azdias.shape[1]) * 100
azdias_nan_row = nan_perc_per_row.sort_values(ascending=False)

missing_threshold = 30
azdias_above_threshold = azdias[azdias_nan_row > missing_threshold]
azdias_below_threshold = azdias[azdias_nan_row <= missing_threshold]

azdias_below_threshold['OST_WEST_KZ'] = azdias_below_threshold['OST_WEST_KZ'].map({'W': 0, 'O': 1})

azdias_below_threshold = azdias_below_threshold.drop(['CAMEO_DEU_2015'], axis=1)
multi_category_col=[]
azdias_below_threshold = pd.get_dummies(azdias_below_threshold, columns = category_multivar)

movement = []
mainstream = [1, 3, 5, 8, 10, 12, 14]
avantgrade = [2, 4, 6, 7, 9, 11, 13, 15]
for row in azdias_below_threshold['PRAEGENDE_JUGENDJAHRE']:
    if row in mainstream:
        movement.append(0)
    elif row in avantgrade:
        movement.append(1)
    else:
        movement.append(np.nan)
azdias_below_threshold['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = np.array(movement)

generation_mapping = {1: 1, 2: 1,
                      3: 2, 4: 2,
                      5: 3, 6: 3,
                      7: 3,
                      8: 4, 9: 4,
                      10: 5, 11: 5, 12: 5, 13: 5,
                      14: 6, 15: 6}
azdias_below_threshold['PRAEGENDE_JUGENDJAHRE_GENERATION'] = azdias_below_threshold['PRAEGENDE_JUGENDJAHRE'].map(generation_mapping)

azdias_below_threshold = azdias_below_threshold.drop('PRAEGENDE_JUGENDJAHRE', axis=1)

cameo_wealth = []
for x in azdias_below_threshold['CAMEO_INTL_2015']:
    if pd.isnull(x):
        cameo_wealth.append(np.nan)
        continue
    x = int(x)
    if 11 <= x <= 15:
        cameo_wealth.append(1)
    elif 21 <= x <= 25:
        cameo_wealth.append(2)
    elif 31 <= x <= 35:
        cameo_wealth.append(3)
    elif 41 <= x <= 45:
        cameo_wealth.append(4)
    elif 51 <= x <= 55:
        cameo_wealth.append(5)
    else:
        cameo_wealth.append(np.nan)
azdias_below_threshold['CAMEO_INTL_2015_WEALTH'] = np.array(cameo_wealth)

cameo_lifestage = []
for x in azdias_below_threshold['CAMEO_INTL_2015']:
    if pd.isnull(x):
        cameo_lifestage.append(np.nan)
        continue
    x = int(x)
    if x % 10 == 1:
        cameo_lifestage.append(1)
    elif x % 10 == 2:
        cameo_lifestage.append(2)
    elif x % 10 == 3:
        cameo_lifestage.append(3)
    elif x % 10 == 4:
        cameo_lifestage.append(4)
    elif x % 10 == 5:
        cameo_lifestage.append(5)
    else:
        cameo_lifestage.append(np.nan)
azdias_below_threshold['CAMEO_INTL_2015_LIFESTAGE'] = np.array(cameo_lifestage)

azdias_below_threshold = azdias_below_threshold.drop('CAMEO_INTL_2015', axis=1)


