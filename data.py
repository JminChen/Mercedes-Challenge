import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

# this file is a preliminary exploration of the training and test data
pd.set_option('mode.chained_assignment', None)


df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
# printing general info on dataset
print(df_train.info())
print(df_test.info())

# finding feature types
dtype_df = df_train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
print(dtype_df)
print(dtype_df.iloc[:10,:])

# finding data raw representation in terms of categorical or numeric or constant
cols = [c for c in df_train.columns if 'X' in c]
counts = [[], [], []]
for c in cols:
    typ = df_train[c].dtype
    uniq = len(np.unique(df_train[c]))
    if uniq == 1: counts[0].append(c)
    elif uniq == 2 and typ == np.int64: counts[1].append(c)
    else: counts[2].append(c)

print('Constant features: {} Binary features: {} Categorical features: {}\n'.format(*[len(c) for c in counts]))

print('Constant features:', counts[0])
print('Categorical features:', counts[2])


#check for missing values
missing_df = df_train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
print(missing_df)

usable_columns = list(set(df_train.columns) - set(['ID', 'y']))

y_train = df_train['y'].values
id_test = df_test['ID'].values

x_train = df_train[usable_columns]
x_test = df_test[usable_columns]


lbl = LabelEncoder()
for column in usable_columns:
    # remove constant values
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)
    if cardinality > 2: # Column is categorical
# convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])


# split features into numeric and categorical
features = df_train.columns[2:]
numeric_features = []
categorical_features = []
for dtype, feature in zip(df_train.dtypes[2:], df_train.columns[2:]):
    if dtype == object:
        categorical_features.append(feature)
    else:
        numeric_features.append(feature)

# find number of unique categorical feature levels in each feature
# observe new representation of categoical data
for col in categorical_features:
    print("Feature {} in train has {} unique values".format(col, len(np.unique(x_train[col]))))
    print(np.unique(x_train[col]))

for col in categorical_features:
    print("Feature {} in test has {} unique values".format(col, len(np.unique(x_test[col]))))
    print(np.unique(x_test[col]))

# observe if there are any categorical values that only appear in the test or training
for col in categorical_features:
    train_less_test = list(set(np.unique(x_train[col])) - set(np.unique(x_test[col])))
    test_less_train = list(set(np.unique(x_test[col])) - set(np.unique(x_train[col])))
    print("For feature {} there are {} training values not in the test\n".format(col, len(train_less_test)))
    print(train_less_test)

    print("And for feature {} there are {} test values not in the train\n".format(col, len(test_less_train)))
    print(test_less_train)
