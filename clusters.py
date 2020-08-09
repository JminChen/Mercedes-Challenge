import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

pd.set_option('mode.chained_assignment', None)

# function to calculate r squared
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

usable_columns = list(set(df_train.columns) - set(['y']))

x_train = df_train[usable_columns]
y_train = df_train['y']
y_mean = np.mean(y_train)

x_test = df_test[usable_columns]

# crate a kmeans model to identify the 4 observed clusters
cluster = KMeans(4,random_state=0)
# group each object in dataset by their X0 values and find the mean y values of those groups
group = df_train.groupby('X0')['y'].mean()
group_2D = group[df_train['X0'].values].values.reshape(-1,1) # change to 2D array to satisfy sklearn KMean
labels = cluster.fit_predict(group_2D) # predict 4 clusters based on the y means of X0 values
temp = pd.DataFrame()
temp['X0'] = df_train['X0']
temp['y'] = df_train['y']
temp['Cluster Label'] = labels
# group each X0 value and find the median of their clusters
cluster_med = temp.groupby('X0')['Cluster Label'].median()
# map each X0 value to a cluster
cluster_train = df_train['X0'].map(cluster_med)
cluster_test = df_test['X0'].map(cluster_med)
# make the cluster label a feature
x_train['Cluster'] = cluster_train
x_test['Cluster'] = cluster_test

lbl = LabelEncoder()
for column in usable_columns:
    if x_train[column].dtypes == 'object': # Column is categorical
    # convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])
    # remove constant features
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)


# parameters for xgboost
xgb_params = {
    # 'n_trees': 520, # max number of trees warned by compiler as unused
    'eta': 0.05, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree kept low to prevent overfitting
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'alpha' : 1, # regularisation parameters
    'lambda' : 2, # regularisation parameters
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:squarederror', # linear regression used as learning task
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
}

# format data to dmatrix for xgboost
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
dtest = xgb.DMatrix(x_test)

# xgboost, cross-validation parameters
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=500, # determines max number of trees to boost and build
                   early_stopping_rounds=50, # if 50 rounds of no improvement then terminate
                   nfold = 10, # nfold the original dataset is randomly partitioned into nfold equal size subsamples.
                   verbose_eval=50,
                   show_stdv=False
                  )

# determine number of boosted rounds by cv
num_boost_rounds = len(cv_result)
print("number of boosted rounds {}" .format(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds, feval=xgb_r2_score, maximize=True)
# use model to predict test set
y_predict = model.predict(dtest)
# print r squared on training data
print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

# print results to csv file called cluster_model.csv
sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('cluster_model.csv', index=False)

# plot the Distribution of target values
sns.distplot(y_train,bins=100,kde=False)
plt.show()

#plot the clustered distribution of the y values
plt.hist(y_train.values[cluster_train==2],bins=100,label='cluster 0')
plt.hist(y_train.values[cluster_train==0],bins=70,label='cluster 1')
plt.hist(y_train.values[cluster_train==1],bins=70,label='cluster 2')
plt.hist(y_train.values[cluster_train==3],bins=70,label='cluster 3')
plt.legend()
plt.title('Target y value clusters determined by X0')
plt.xlim((60,170))
plt.show()


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
