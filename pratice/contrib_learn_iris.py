from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn import datasets
import os

import tensorflow as tf
import numpy as np
import pandas as pd


print(os.listdir("D:\kaggle"))
path = 'D:\kaggle\Iris.csv'
iris_data = pd.read_csv(path)
iris_data.head()
iris_data['Species']
znp.astype(iris_data['Species'])
iris_data.describe()
X.unique()

y = iris.target
X

iris_data.head()
X = iris.data[:, [2, 3]]
y = iris.target

# Place the iris data into a pandas dataframe
iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])
iris_df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))
print()
https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset
