# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint
df = pd.read_csv('/content/drive/MyDrive/data/knn_data.csv', encoding='utf-8')

print(df)

from sklearn.preprocessing import MinMaxScaler
#do minmax scaling
scaler = MinMaxScaler()

df[['longitude', 'latitude']] = scaler.fit_transform(df[['longitude', 'latitude']])
print(df[['longitude', 'latitude']])

X = df[['longitude', 'latitude']] #feature column
y = df['lang'] #target

#encode target data
le = LabelEncoder()
y = le.fit_transform(y)

# Cross-validation
scores = cross_val_score(KNeighborsClassifier(), X, y, cv=5)
print("Cross-validation scores: {}".format(scores))

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Instantiate the grid search
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# Fit the model
grid_search.fit(X, y)
print("GridSearchCV")
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Instantiate the randomized search
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# Fit the model
random_search.fit(X, y)

print()
print("RandomizedSearch")
print("Best parameters : {}".format(random_search.best_params_))
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))