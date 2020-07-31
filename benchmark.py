import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import r2_score

# color = sns.color_palette()

from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")

usable_columns = list(set(df_train.columns) - set(['ID', 'y']))

x_train = df_train[usable_columns]
y_train = df_train['y'].values

x_test = df_test[usable_columns]

# remove constant values
lbl = LabelEncoder()
for column in usable_columns:
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1, inplace=True) # Column with only one value is useless so we drop it
        x_test.drop(column, axis=1, inplace=True)
    if cardinality > 2: # Column is categorical
    # need to find a good cateogorical to numerical mapper
# convert categorical features into numeric
        x_train[column] = lbl.fit_transform(x_train[column])
        x_test[column] = lbl.fit_transform(x_test[column])

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,model.predict(dtrain)))

# plot the Distribution of target values
sns.distplot(y_train[y_train<170],bins=100,kde=False)
plt.show()

# plot the important features #
# fig, ax = plt.subplots(figsize=(12,18))
# xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
# plt.show()
