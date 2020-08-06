import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

import pylab

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

# usable_columns = list(set(df_train.columns) - set(['ID', 'y']))
usable_columns = list(set(df_train.columns) - set(['y']))

x_train = df_train[usable_columns]
y_train = df_train['y']
y_mean = np.mean(y_train)

x_test = df_test[usable_columns]

cluster = KMeans(4,random_state=0)
# group each object in dataset by their X0 values and find the mean y values of those groups
group = df_train.groupby('X0')['y'].mean()
group_2D = group[df_train['X0'].values].values.reshape(-1,1) # change to 2D array to satisfy sklearn KMean
labels = cluster.fit_predict(group_2D) # predict 4 clusters based y means of X0 values
temp = pd.DataFrame()
temp['X0'] = df_train['X0'].values
temp['Cluster Label'] = labels
cluster_med = temp.groupby('X0')['Cluster Label'].median() # group each X0 value and find the median of their clusters

cluster_train = df_train['X0'].map(cluster_med)
cluster_test = df_test['X0'].map(cluster_med)

# df_train['Cluster'] = cluster_train
# df_test['Cluster'] = cluster_test
x_train['Cluster'] = cluster_train
x_test['Cluster'] = cluster_test
# print(x_test['Cluster'])

# est = xgb.XGBClassifier()
# est.fit(x_train.select_dtypes(include=[np.number]),cluster_train)
# cluster_test[np.isnan(cluster_test)] = est.predict(
#     x_test.select_dtypes(include=[np.number]))[np.isnan(cluster_test)]

# plt.hist(y_train.values[cluster_train==2],bins=70,label='cluster 0')
# plt.hist(y_train.values[cluster_train==0],bins=70,label='cluster 1')
# plt.hist(y_train.values[cluster_train==1],bins=100,label='cluster 2')
# plt.hist(y_train.values[cluster_train==3],bins=70,label='cluster 3')
# plt.legend()
# plt.title('Train targets distribution for all clusters')
# plt.xlim((60,170))
# plt.show()

lbl = LabelEncoder()
for column in usable_columns:
    if x_train[column].dtypes == 'object': # Column is categorical
    # need to find a good cateogorical to numerical mapper
    # convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])

        # lbl.fit(list(x_train[column].values) + list(x_test[column].values))
        # x_train[column] = lbl.transform(list(x_train[column].values))
        # x_test[column] = lbl.transform(list(x_test[column].values))

    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)

#colsample_bytree
# maybe split data according to clusters and train model on each cluster

kb = SelectKBest(score_func=f_regression, k=200)
kb.fit(x_train,y_train)
best_train = kb.transform(x_train)
# CONTAINS NAN VALUES PROBABLY FROM objets in test set that couldnt be assigned to a cluster
# print(x_train['Cluster'].isnull().values.any())
# print(x_test['Cluster'].isnull().sum())
x_test.fillna(4,inplace=True)
best_test = kb.transform(x_test)

# xgboost used since excels at small-medium sized tabular data and regression prediction
# perhaps try other models random forest,
xgb_params = {
    'n_trees': 520, # max number of trees
    'eta': 0.05, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree kept low to prevent overfitting
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:linear', # linear regression used as learning task
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
    'silent': 1
}

# #split data
# c1 = df_train[df_train['Cluster']==0]
# c2 = df_train.loc[df_train['Cluster']==1]
# c3 = df_train.loc[df_train['Cluster']==2]
# c4 = df_train.loc[df_train['Cluster']==3]
#
# x1 = c1[list(set(c1.columns) - set(['y']))]
# y1 = c1['y']
# x2 = c2[list(set(c2.columns) - set(['y']))]
# y2 = c2['y']
# x3 = c3[list(set(c3.columns) - set(['y']))]
# y3 = c3['y']
# x4 = c4[list(set(c4.columns) - set(['y']))]
# y4 = c4['y']
#
# x1_test = df_test[df_test['Cluster']==0]
# print(x1_test)
# ID1 = x1_test['ID']
# x2_test = df_test.loc[df_test['Cluster']==1]
# x3_test = df_test.loc[df_test['Cluster']==2]
# x4_test = df_test.loc[df_test['Cluster']==3]

# x1_test = df_train.loc[df_test['Cluster']==0]
# x2_test = df_train.loc[df_test['Cluster']==1]
# x3_test = df_train.loc[df_test['Cluster']==2]
# x4_test = df_train.loc[df_test['Cluster']==3]

#
# dtrain = xgb.DMatrix(x1, y1,feature_names=x1.columns.values)
# dtest = xgb.DMatrix(x1_test)

# dtrain1 = xgb.DMatrix(x1, y1)
# dtrain1 = xgb.DMatrix(x1, y1)
# dtrain1 = xgb.DMatrix(x1, y1)

dtrain = xgb.DMatrix(best_train, y_train)
dtest = xgb.DMatrix(best_test)

# dtrain = xgb.DMatrix(df_train, y_train, feature_names=x_train.columns.values)
# dtest = xgb.DMatrix(x_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=500, # determines number of trees to boost and build, considering increasing
                   early_stopping_rounds=50, # if 50 rounds of no improvement then terminate
                   nfold = 10, # nfold  the original dataset is randomly partitioned into nfold equal size subsamples.
                   #seed = 42, # random seed
                   verbose_eval=50,
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)


model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, feval=xgb_r2_score, maximize=True)
y_predict = model.predict(dtest)


# num_boost_rounds = len(cv_result)
# print("number of rounds {}" .format(num_boost_rounds))
#
# # dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
#
# # dtest = xgb.DMatrix(x_test)
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, feval=xgb_r2_score, maximize=True)
#
y_predict = model.predict(dtest)
'''R2 Score on the entire Train data'''

print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('model.csv', index=False)

# # # plot the Distribution of target values
# # sns.distplot(y_train[y_train<170],bins=100,kde=False)
# # plt.show()
#
# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
