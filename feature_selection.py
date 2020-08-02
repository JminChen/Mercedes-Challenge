import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import r2_score
from numpy import sort

from sklearn.feature_selection import SelectFromModel

# color = sns.color_palette()

from sklearn.preprocessing import LabelEncoder

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

# remove constant values
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



def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

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

dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
dtest = xgb.DMatrix(x_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50,
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print("number of rounds {}" .format(num_boost_rounds))

# dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)

# dtest = xgb.DMatrix(x_test)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, feval=xgb_r2_score, maximize=True)

# thresholds = sorted(model.get_score(importance_type='gain').items(), key=lambda x: x[1])
# # print(model.get_score(importance_type = 'gain')
#
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(x_train)
#     # train model
#     dtrain = xgb.DMatrix(select_X_train, y_train, feature_names=select_X_train.columns.values)
#     selection_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
#
#     # eval model
#     select_X_test = selection.transform(x_test)
#     dtest = xgb.DMatrix(select_X_test)
#     y_predict = selection_model.predict(dtest)

y_predict = model.predict(dtest)
'''R2 Score on the entire Train data'''

print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))



sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('model.csv', index=False)

# # plot the Distribution of target values
# sns.distplot(y_train[y_train<170],bins=100,kde=False)
# plt.show()



# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
