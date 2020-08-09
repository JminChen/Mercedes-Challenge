# Mercedes-Challenge
## Introduction

The following files are the models we used to attempt the Mercedes Greener Manufacturing challenge on Kaggle.

## Installation

The following files require pip install of numerous packages including numpy, pandas, sklearn, matplotlib, seaborn as well as xgboost.
```bash
pip install numpy
pip install sklearn
pip install pandas
pip install sklearn
pip install matplotlib
pip install seaborn
pip install xgboost

```

## Details about files
### train.csv
The training data set with target values.

### test.csv
The test data set with target values removed.

### data.py
A preliminary exploration of the training and test set, including determining which features were categorical, binary or constant.

### feature_distribution.py
A file used to print the values and distribution of our 8 categorical features.

### simpleLinear.py
Our initial primitive linear regression model we used to try and solve the problem.

### baseline.py  
Our baseline model created using xgboost which we based successive models on.

### cross_val.py
Used xgboosts cross validation package to determine the number of boosted rounds, we discovered this improved our baseline results

### feature_selection.py
An attempt to improve the baseline model with cross validation by implementing select k best feature selection to try and eliminate noise on the dataset and improve our results. We also attempted using random forests to select features.

### dim_red_substitute.py
A more sophisticated attempt at improving our xgboost model using feature selection by implementing dimensionality reduction techniques

### clusters.py
Our final model using xgboost, cross validation and k means clustering to map the distributions in our data.

### any.csv file
These are the target values of the test data predicted by the respective models. To evaluate the R squared or result of the test set, you are required to submit this results file to the Kaggle competition leaderboard.
