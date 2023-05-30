
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
warnings.filterwarnings(action="ignore")
from sklearn.datasets import load_iris

#load iris data
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())

X = df.loc[:, :'petal width (cm)'] #
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Train a DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# do bagging
bagging = BaggingClassifier(tree, n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)

# Predict the labels using voting
y_pred = bagging.predict(X_test)

# Calculate accuracy using confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_mat)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

