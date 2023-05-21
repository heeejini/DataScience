# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import warnings
warnings.filterwarnings(action='ignore')
from pandas import Series, DataFrame
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/drive/MyDrive/data/decision_tree_data.csv', encoding='utf-8')
print(df.head())

#encoding
df_encoded = pd.get_dummies(df, drop_first=False)

#the encoded DataFrame
print(df_encoded.head())

X = df_encoded.drop("interview", axis=1)
y = df_encoded["interview"]

# Define the train-test split size
split_sizes = [0.1,0.2,0.3]

for size in split_sizes:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, shuffle=True,random_state=42, stratify=y
    )

    # Train the decision tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Training size: {:.2f}, Testing size: {:.2f}".format(1- size,size))
    print("Accuracy: {:.4f}".format(accuracy))
    print()