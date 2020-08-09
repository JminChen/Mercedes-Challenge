import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder

pd.set_option('mode.chained_assignment', None)

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

# we decided to include ID as a feature
usable_columns = list(set(df_train.columns) - set(['y']))
# usable_columns = list(set(df_train.columns) - set(['ID', 'y']))

x_train = df_train[usable_columns]
y_train = df_train['y'].values
y_mean = np.mean(y_train)

x_test = df_test[usable_columns]

#feature engineering
lbl = LabelEncoder()
for column in usable_columns:
    if x_train[column].dtypes == 'object': # Column is categorical
        # convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])

    # remove features with 0 variation
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)


# function to calculate r squared
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

# parameters for our xgboost model
xgb_params = {
    # 'n_trees': 520, # max number of trees realised this value was never used
    'eta': 0.05, # learning rate kept low to prevent overfitting
    # noted when eta was 0.0045 r squared decreased substantially
    'max_depth': 5, # deepest leaf of tree kept low to prevent overfitting
    'alpha' : 1, # regularisation parameters
    'lambda' : 2, # regularisation parameters
    'subsample': 0.7, # percentage of data used in each bunch of trees
    'colsample_bytree': 0.7, # percentage of feature columns used in each tree
    'objective': 'reg:squarederror', # squared error regression used as learning task
    'eval_metric': 'rmse', # root mean squared error used as loss function for linear regression
    'base_score': y_mean, # sets initial prediction of all instances to mean of dataset
    # 'silent': 1
}

# format so xgboost can read
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
dtest = xgb.DMatrix(x_test)
# train model on training set using the parameters specified
model = xgb.train(dict(xgb_params), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
# use model ot predict training set
y_predict = model.predict(dtest)

# finding r2 score on train data
print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

# print predictions to csv file
sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['y'] = y_predict
sub.to_csv('baseline_model.csv', index=False)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
