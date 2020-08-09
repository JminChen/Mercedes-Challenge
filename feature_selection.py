import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import r2_score
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder

pd.set_option('mode.chained_assignment', None)
# read training and test data
df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
usable_columns = list(set(df_train.columns) - set(['y']))
x_train = df_train[usable_columns]
y_train = df_train['y'].values
y_mean = np.mean(y_train)
x_test = df_test[usable_columns]


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


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

# select top 200 features according to importance score
kb = SelectKBest(score_func=f_regression, k=200)
kb.fit(x_train,y_train)
best_train = kb.transform(x_train)
best_test = kb.transform(x_test)

# attempted to perform feature selection using random forest regression and select k best but resulted
# in very poor r squared and didn't improve our base model

# slt = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=20, n_jobs=-1)
# slt.fit(x_train, y_train)
# select = SelectFromModel(slt)
# select.fit(x_train, y_train)
# # selected_features= x_train.columns[(select.get_support())]
# important_feat_train = select.transform(x_train)
# important_feat_test = select.transform(x_test)
#
# # then select top 40 features
# kb = SelectKBest(score_func=f_regression, k=40)
# kb.fit(important_feat_train,y_train)
# best_train = kb.transform(important_feat_train)
# best_test = kb.transform(important_feat_test)


xgb_params = {
    'n_trees': 520, # max number of trees
    'eta': 0.05, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree kept low to prevent overfitting
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:linear', # linear regression used as learning task
    'alpha' : 1, # regularisation parameters
    'lambda' : 2, # regularisation parameters
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
    'silent': 1
}

dtrain = xgb.DMatrix(best_train, y_train)
dtest = xgb.DMatrix(best_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=500, # determines number of trees to boost and build, considering increasing
                   early_stopping_rounds=50, # if 50 rounds of no improvement then terminate
                   verbose_eval=50,
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print("number of rounds {}" .format(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, feval=xgb_r2_score, maximize=True)
# use model to predict test set
y_predict = model.predict(dtest)
print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

# print test prediction to a csv file
sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('feature_selection_model.csv', index=False)


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
