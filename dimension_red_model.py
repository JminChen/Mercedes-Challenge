import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration

# color = sns.color_palette()

from sklearn.preprocessing import LabelEncoder

# function for finding r squared
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

# usable_columns = list(set(df_train.columns) - set(['ID', 'y']))
usable_columns = list(set(df_train.columns) - set(['y']))

x_train = df_train[usable_columns]
y_train = df_train['y'].values
y_mean = np.mean(y_train)

x_test = df_test[usable_columns]
# test set no y values
# y_test = df_test['y'].values

# do some feature engineering
lbl = LabelEncoder()
for column in usable_columns:
    if x_train[column].dtypes == 'object': # Column is categorical
    # need to find a good cateogorical to numerical mapper
    # convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])
    # remove constant features
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)


#before model perform some dimensionality reduction

# PCA
remaining_comp = 15 # number of dimensions the data is reduced to
pca = PCA(n_components=remaining_comp, random_state=420)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

# tSVD
tsvd = TruncatedSVD(n_components=remaining_comp, random_state=420)
tsvd_train = tsvd.fit_transform(x_train)
tsvd_test = tsvd.transform(x_test)

# ICA
ica = FastICA(n_components=remaining_comp, random_state=420)
ica_train = ica.fit_transform(x_train)
ica_test = ica.transform(x_test)

# GRP
grp = GaussianRandomProjection(n_components=remaining_comp, eps=0.1, random_state=420)
grp_train = grp.fit_transform(x_train)
grp_test = grp.transform(x_test)

# SRP
srp = SparseRandomProjection(n_components=remaining_comp, dense_output=True, random_state=420)
srp_train = srp.fit_transform(x_train)
srp_test = srp.transform(x_test)

# # NMF
# nmf = NMF(n_components=remaining_comp, init='nndsvdar', random_state=420)
# nmf_train = nmf.fit_transform(x_train)
# nmf_test = nmf.transform(x_test)
#
# # FAG
# fag = FeatureAgglomeration(n_clusters=remaining_comp, linkage='ward')
# fag_train = fag.fit_transform(x_train)
# fag_test = fag.transform(x_test)

for i in range(1, remaining_comp + 1):
    x_train['pca_' + str(i)] = pca_train[:, i - 1]
    x_test['pca_' + str(i)] = pca_test[:, i - 1]

    x_train['ica_' + str(i)] = ica_train[:, i - 1]
    x_test['ica_' + str(i)] = ica_test[:, i - 1]

    x_train['tsvd_' + str(i)] = tsvd_train[:, i - 1]
    x_test['tsvd_' + str(i)] = tsvd_test[:, i - 1]

    x_train['grp_' + str(i)] = grp_train[:, i - 1]
    x_test['grp_' + str(i)] = grp_test[:, i - 1]

    x_train['srp_' + str(i)] = srp_train[:, i - 1]
    x_test['srp_' + str(i)] = srp_test[:, i - 1]
    #
    # x_train['nmf_' + str(i)] = nmf_train[:, i - 1]
    # x_test['nmf_' + str(i)] = nmf_test[:, i - 1]
    #
    # x_train['fag_' + str(i)] = fag_train[:, i - 1]
    # x_test['fag_' + str(i)] = fag_test[:, i - 1]


# xgboost used since excels at small-medium sized tabular data and regression prediction
# perhaps try other models random forest,
xgb_params = {
    'n_trees': 520, # max number of trees
    'eta': 0.045, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree kept low to prevent overfitting
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:linear', # linear regression used as learning task
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
}

# then perform feature selection

dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)

dtest = xgb.DMatrix(x_test)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

y_predict = model.predict(dtest)
'''R2 Score on the entire Train data'''

print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

# '''R2 Score on the Test data'''
#
# print('R2 score on test data:')
# print(r2_score(y_test,y_predict))


sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('model.csv', index=False)

# plot the Distribution of target values
# sns.distplot(y_train[y_train<170],bins=100,kde=False)
# plt.show()



# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
