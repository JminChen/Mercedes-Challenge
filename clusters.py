import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import LabelEncoder

import pylab
df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

# usable_columns = list(set(df_train.columns) - set(['ID', 'y']))
usable_columns = list(set(df_train.columns) - set(['y']))

x_train = df_train[usable_columns]
y_train = df_train['y']
y_mean = np.mean(y_train)

x_test = df_test[usable_columns]

lbl = LabelEncoder()
for column in usable_columns:
    # if x_train[column].dtypes == 'object': # Column is categorical
    # # need to find a good cateogorical to numerical mapper
    # # convert categorical features into numeric
    #     x_train[column] = lbl.fit_transform(x_train[column])
    #     x_test[column] = lbl.fit_transform(x_test[column])
    #     # lbl.fit(list(x_train[column].values) + list(x_test[column].values))
    #     # x_train[column] = lbl.transform(list(x_train[column].values))
    #     # x_test[column] = lbl.transform(list(x_test[column].values))

    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)

# class cluster_target_encoder:
#     def make_encoding(self,df):
#         self.encoding = df.groupby('X')['y'].mean()
#     def fit(self,X,y):
#         df = pd.DataFrame(columns=['X','y'],index=X.index)
#         df['X'] = X
#         df['y'] = y
#         self.make_encoding(df)
#         clust = KMeans(4,random_state=0)
#         labels = clust.fit_predict(self.encoding[df['X'].values].values.reshape(-1,1))
#         df['labels'] = labels
#         self.clust_encoding = df.groupby('X')['labels'].median()
#     def transform(self,X):
#         res = X.map(self.clust_encoding).astype(float)
#         return res
#     def fit_transform(self,X,y):
#         self.fit(X,y)
#         return self.transform(X)
#
#
# encoder = cluster_target_encoder(nclusters=4,seed=4)
# labels_train = encoder.fit_transform(x_train['X0'],train['y'])
# labels_test = encoder.transform(X_test['X0'])
# est = xgb.XGBClassifier()
# est.fit(x_train.select_dtypes(include=[np.number]),labels_train)
# labels_test[np.isnan(labels_test)] = est.predict(
#     x_test.select_dtypes(include=[np.number]))[np.isnan(labels_test)]
# x_train['labels'] = labels_train
# x_test['labels'] = labels_test

cluster = KMeans(4,random_state=0)
# group each object in dataset by their X0 values and find the mean y values of those groups
group = df_train.groupby('X0')['y'].mean()
group_2D = group[df_train['X0'].values].values.reshape(-1,1) # change to 2D array to satisfy sklearn KMean
labels = cluster.fit_predict(group_2D) # predict 4 clusters
temp = pd.DataFrame()
temp['X0'] = df_train['X0'].values
temp['Cluster Label'] = labels
cluster_med = temp.groupby('X0')['Cluster Label'].median()

cluster_train = df_train['X0'].map(cluster_med).astype(float)
cluster_test = df_test['X0'].map(cluster_med).astype(float)

# # print(group)
# # clust.fit_predict()

plt.hist(y_train.values[cluster_train==2],bins=70,label='cluster 0')
plt.hist(y_train.values[cluster_train==0],bins=70,label='cluster 1')
plt.hist(y_train.values[cluster_train==1],bins=100,label='cluster 2')
plt.hist(y_train.values[cluster_train==3],bins=70,label='cluster 3')
plt.legend()
plt.title('Train targets distribution for all clusters')
plt.xlim((60,170))
plt.show()



# # plot the Distribution of target values
# sns.distplot(y_train[y_train<170],bins=100,kde=False)
# plt.show()
