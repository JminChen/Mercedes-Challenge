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

x_test = df_test[usable_columns]

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

remaining_comp = 50 # number of dimensions the data is reduced to

# PCA
pca = PCA(n_components=30, random_state=420)
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
feag = FeatureAgglomeration(n_clusters=30, linkage='ward')
feag_xtrain = feag.fit_transform(x_train)
feag_xtest = feag.transform(x_test)

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

# Basic model
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
dtest = xgb.DMatrix(x_test)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
y_predict = model.predict(dtest)

# Test model on PCA dataset
PCAtrain = xgb.DMatrix(pca_xtrain, y_train)
PCAtest = xgb.DMatrix(pca_xtest)
PCA_model = xgb.train(dict(xgb_params), PCAtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
PCA_ypredict = PCA_model.predict(PCAtest)

pca_out = pd.DataFrame()
pca_out['ID'] = df_test['ID']
pca_out['y'] = PCA_ypredict
pca_out.to_csv('pca_data.csv', index=False)


# Test model on TruncatedSVD dataset
TSVDtrain = xgb.DMatrix(tsvd_xtrain, y_train)
TSVDtest = xgb.DMatrix(tsvd_xtest)
TSVD_model = xgb.train(dict(xgb_params), TSVDtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
TSVD_ypredict = TSVD_model.predict(TSVDtest)

tsvd_out = pd.DataFrame()
tsvd_out['ID'] = df_test['ID']
tsvd_out['y'] = TSVD_ypredict
tsvd_out.to_csv('tsvd_data.csv', index=False)

# Test model on ICA dataset
ICAtrain = xgb.DMatrix(ica_xtrain, y_train)
ICAtest = xgb.DMatrix(ica_xtest)
ICA_model = xgb.train(dict(xgb_params), ICAtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
ICA_ypredict = ICA_model.predict(ICAtest)

ica_out = pd.DataFrame()
ica_out['ID'] = df_test['ID']
ica_out['y'] = ICA_ypredict
ica_out.to_csv('ica_data.csv', index=False)


# Test model on GRP dataset
GRPtrain = xgb.DMatrix(grp_xtrain, y_train)
GRPtest = xgb.DMatrix(grp_xtest)
GRP_model = xgb.train(dict(xgb_params), GRPtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
GRP_ypredict = GRP_model.predict(GRPtest)



# Test model on SRP dataset
SRPtrain = xgb.DMatrix(srp_xtrain, y_train)
SRPtest = xgb.DMatrix(srp_xtest)
SRP_model = xgb.train(dict(xgb_params), SRPtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
SRP_ypredict = SRP_model.predict(SRPtest)



# Test model on NMF dataset
NMFtrain = xgb.DMatrix(nmf_xtrain, y_train)
NMFtest = xgb.DMatrix(nmf_xtest)
NMF_model = xgb.train(dict(xgb_params), NMFtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
NMF_ypredict = NMF_model.predict(NMFtest)



# Test model on TruncatedSVD dataset
FEAGtrain = xgb.DMatrix(feag_xtrain, y_train)
FEAGtest = xgb.DMatrix(feag_xtest)
FEAG_model = xgb.train(dict(xgb_params), FEAGtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
FEAG_ypredict = FEAG_model.predict(FEAGtest)

feag_out = pd.DataFrame()
feag_out['ID'] = df_test['ID']
feag_out['y'] = FEAG_ypredict
feag_out.to_csv('feag_data.csv', index=False)

# plot the Distribution of target values
# sns.distplot(y_train[y_train<170],bins=100,kde=False)
# plt.show()

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(FEAG_model, max_num_features=100, height=0.8, ax=ax)
plt.show()

# xgb.plot_importance(TSVD_model, max_num_features=50, height=0.8, ax=ax)
# plt.show()

# xgb.plot_importance(ICA_model, max_num_features=50, height=0.8, ax=ax)
# plt.show()
#
