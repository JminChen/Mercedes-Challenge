from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# test and training data csv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# LabelEncoding for categorical features
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))

train_y = train['y'].values
train_X = train.drop(["ID", "y"], axis=1)

train_X = train_X.values;

# PCA transform
components = 20
pca = PCA(n_components = components)
train_X = pca.fit_transform(train_X)

# 10-fold Cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
lr = LinearRegression()

for train_index, test_index in kf.split(train_X):
    X_train, X_test=train_X[train_index], train_X[test_index]
    y_train, y_test=train_y[train_index], train_y[test_index]
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))

for f in ["X0",  "X1",  "X2",  "X3",  "X4",  "X5",  "X6",  "X8"]:
        lbl=preprocessing.LabelEncoder()
        # fit data value
        lbl.fit(list(test[f].values))
        test[f]=lbl.transform(list(test[f].values))

X_test=test.drop(["ID"], axis=1).values
X_test=pca.fit_transform(X_test)
y_test=lr.predict(X_test)

# Submit into csv file
id_list=test['ID'].values
sub = pd.DataFrame(

    {'ID': id_list,
     'y': y_test,
    })

with open('SimpleLinear.csv', 'w') as f:
    sub.to_csv(f, index=False)
