import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration
import warnings
warnings.filterwarnings(action='ignore')

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

# x_test = df_test[usable_columns]
# test set no y values
# y_test = df_test['y'].values

# Resample the data with replacement to test
x_test, y_test = resample(x_train, y_train, random_state=100)

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

remaining_comp = 80 # number of dimensions the data is reduced to

# PCA
pca = PCA(n_components=remaining_comp, random_state=420)
pca_xtrain = pca.fit_transform(x_train)
pca_xtest = pca.transform(x_test)

# tSVD
tsvd = TruncatedSVD(n_components=remaining_comp, random_state=420)
tsvd_xtrain = tsvd.fit_transform(x_train)
tsvd_xtest = tsvd.transform(x_test)

# ICA
ica = FastICA(n_components=remaining_comp, random_state=420)
ica_xtrain = ica.fit_transform(x_train)
ica_xtest = ica.transform(x_test)

# GRP
grp = GaussianRandomProjection(n_components=remaining_comp, eps=0.1, random_state=420)
grp_xtrain = grp.fit_transform(x_train)
grp_xtest = grp.transform(x_test)

# SRP
srp = SparseRandomProjection(n_components=remaining_comp, dense_output=True, random_state=420)
srp_xtrain = srp.fit_transform(x_train)
srp_xtest = srp.transform(x_test)

# NMF
nmf = NMF(n_components=remaining_comp, init='nndsvdar', random_state=420)
nmf_xtrain = nmf.fit_transform(x_train)
nmf_xtest = nmf.transform(x_test)

# FEAG
feag = FeatureAgglomeration(n_clusters=remaining_comp, linkage='ward')
feag_xtrain = feag.fit_transform(x_train)
feag_xtest = feag.transform(x_test)

# for i in range(1, remaining_comp + 1):
#     x_train['pca_' + str(i)] = pca_train[:, i - 1]
#     x_test['pca_' + str(i)] = pca_test[:, i - 1]
#
#     x_train['ica_' + str(i)] = ica_train[:, i - 1]
#     x_test['ica_' + str(i)] = ica_test[:, i - 1]
#
#     x_train['tsvd_' + str(i)] = tsvd_train[:, i - 1]
#     x_test['tsvd_' + str(i)] = tsvd_test[:, i - 1]
#
#     x_train['grp_' + str(i)] = grp_train[:, i - 1]
#     x_test['grp_' + str(i)] = grp_test[:, i - 1]
#
#     x_train['srp_' + str(i)] = srp_train[:, i - 1]
#     x_test['srp_' + str(i)] = srp_test[:, i - 1]
#
#     x_train['nmf_' + str(i)] = nmf_train[:, i - 1]
#     x_test['nmf_' + str(i)] = nmf_test[:, i - 1]
#
#     x_train['fag_' + str(i)] = fag_train[:, i - 1]
#     x_test['fag_' + str(i)] = fag_test[:, i - 1]

# reduced = pd.DataFrame()
# test_reduced = pd.DataFrame()
# for i in range(1, remaining_comp + 1):
#     reduced['pca_' + str(i)] = pca_train[:, i - 1]
#     # reduced['ica_' + str(i)] = ica_train[:, i - 1]
#     # reduced['tsvd_' + str(i)] = tsvd_train[:, i - 1]
#     # reduced['grp_' + str(i)] = grp_train[:, i - 1]
#     # reduced['srp_' + str(i)] = srp_train[:, i - 1]
#     # # reduced['nmf_' + str(i)] = nmf_train[:, i - 1]
#     # # reduced['fag_' + str(i)] = fag_train[:, i - 1]
#
#     test_reduced['pca_' + str(i)] = pca_test[:, i - 1]
#     # test_reduced['ica_' + str(i)] = ica_train[:, i - 1]
#     # test_reduced['tsvd_' + str(i)] = tsvd_train[:, i - 1]
#     # test_reduced['grp_' + str(i)] = grp_train[:, i - 1]
#     # test_reduced['srp_' + str(i)] = srp_train[:, i - 1]
#     # # test_reduced['nmf_' + str(i)] = nmf_train[:, i - 1]
#     # # test_reduced['fag_' + str(i)] = fag_train[:, i - 1]

# xgboost used since excels at small-medium sized tabular data and regression prediction
# perhaps try other models random forest,
xgb_params = {
    #'n_trees': 520, # max number of trees
    'eta': 0.05, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:squarederror', # squared error used as learning task
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
}

# then perform feature selection

# dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)

# rtest = xgb.DMatrix(test_reduced, feature_names = test_reduced.columns.values)
# # model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
#
# rtrain = xgb.DMatrix(reduced,y_train, feature_names = reduced.columns.values)
# model = xgb.train(dict(xgb_params, silent=0), rtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
#
# y_predict = model.predict(rtest)

# Basic model
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
dtest = xgb.DMatrix(x_test)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
y_predict = model.predict(dtest)

# '''R2 Score on the entire Train data'''
# print('R2 score on train data:')
# print(r2_score(y_train,model.predict(dtrain)))
'''R2 Score on the Test data'''
print('Base R2 score on test data:')
print(r2_score(y_test,y_predict))
print('')

# Test model on PCA dataset
PCAtrain = xgb.DMatrix(pca_xtrain, y_train)
PCAtest = xgb.DMatrix(pca_xtest)
model = xgb.train(dict(xgb_params), PCAtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
PCA_ypredict = model.predict(PCAtest)
# Print R^2 score
print('Principle Component Analysis R2 score:')
print(r2_score(y_test,PCA_ypredict))
print('')


# Test model on TruncatedSVD dataset
TSVDtrain = xgb.DMatrix(tsvd_xtrain, y_train)
TSVDtest = xgb.DMatrix(tsvd_xtest)
model = xgb.train(dict(xgb_params), TSVDtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
TSVD_ypredict = model.predict(TSVDtest)
# Print R^2 score
print('Truncated Singular Value Decomposition R2 score:')
print(r2_score(y_test,TSVD_ypredict))
print('')


# Test model on ICA dataset
ICAtrain = xgb.DMatrix(ica_xtrain, y_train)
ICAtest = xgb.DMatrix(ica_xtest)
model = xgb.train(dict(xgb_params), ICAtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
ICA_ypredict = model.predict(ICAtest)
# Print R^2 score
print('Independant Component Analysis R2 score:')
print(r2_score(y_test,ICA_ypredict))
print('')


# Test model on GRP dataset
GRPtrain = xgb.DMatrix(grp_xtrain, y_train)
GRPtest = xgb.DMatrix(grp_xtest)
model = xgb.train(dict(xgb_params), GRPtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
GRP_ypredict = model.predict(GRPtest)
# Print R^2 score
print('Gaussian Random Projection R2 score:')
print(r2_score(y_test,GRP_ypredict))
print('')


# Test model on SRP dataset
SRPtrain = xgb.DMatrix(srp_xtrain, y_train)
SRPtest = xgb.DMatrix(srp_xtest)
model = xgb.train(dict(xgb_params), SRPtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
SRP_ypredict = model.predict(SRPtest)
# Print R^2 score
print('Sparse Random Projection R2 score:')
print(r2_score(y_test,SRP_ypredict))
print('')


# Test model on NMF dataset
NMFtrain = xgb.DMatrix(nmf_xtrain, y_train)
NMFtest = xgb.DMatrix(nmf_xtest)
model = xgb.train(dict(xgb_params), NMFtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
NMF_ypredict = model.predict(NMFtest)
# Print R^2 score
print('Non-negative Matrix Factorisation R2 score:')
print(r2_score(y_test,NMF_ypredict))
print('')


# Test model on TruncatedSVD dataset
FEAGtrain = xgb.DMatrix(feag_xtrain, y_train)
FEAGtest = xgb.DMatrix(feag_xtest)
model = xgb.train(dict(xgb_params), FEAGtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
FEAG_ypredict = model.predict(FEAGtest)
# Print R^2 score
print('Feature Agglomeration R2 score:')
print(r2_score(y_test,FEAG_ypredict))
print('')



# '''Average the preditionon test data  of both models then save it on a csv file'''

# sub = pd.DataFrame()
# sub['ID'] = df_test['ID']
# sub['y'] = y_predict
# sub.to_csv('model.csv', index=False)

# plot the Distribution of target values
# sns.distplot(y_train[y_train<170],bins=100,kde=False)
# plt.show()

# plot the important features #
# fig, ax = plt.subplots(figsize=(12,18))
# xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
# plt.show()
