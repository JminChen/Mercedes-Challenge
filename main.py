import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# import xgboost as xgb
# color = sns.color_palette()

# %matplotlib inline

# pd.options.mode.chained_assignment = None  # default='warn'
# pd.options.display.max_columns = 999

def main():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

if __name__ == "__main__":
    main()
