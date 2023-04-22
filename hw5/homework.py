# -*- coding: utf-8 -*-
"""
using scikit-learn and seaborn library, find the regression line and also draw the line and a scatter plot of the dataset
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection

data = {'spends' :[2400,2650,2350,4950,3100,2500,5106,3100,2900,1750],
        'income' : [41200,50100,52000,66000,44500,37700,73500,37500,56700,35600]}


df = pd.DataFrame(data,columns=['spends','income'])
#print(df)

x=pd.DataFrame(df['spends'])
y=pd.DataFrame(df['income'])
#print(x , "\n" ,y)

model = linear_model.LinearRegression()

scores = []
kfold = model_selection.KFold(n_splits=3, shuffle=True, random_state=10)

for i, (train, test) in enumerate(kfold.split(x,y)):
  model.fit(x.iloc[train,:], y.iloc[train,:])
  score = model.score(x.iloc[test,:],y.iloc[test,:])
  scores.append(score)

print(scores)

#using sciklearn

#convert dataframe to numpy
X = x.to_numpy()
#print(X)
Y =y.to_numpy()
#print(Y)

reg = linear_model.LinearRegression()
reg.fit(X,Y)
# fit linear model

px = np.array([X.min()-1000, X.max()+1000])
py = reg.predict(px[:, np.newaxis])

plt.axis([1500,5500,30000,75000]) #Setting the range of the x and y axis
plt.title("Linear Regression with scikit-learn")
plt.scatter(X, Y, color='r')
plt.plot(px, py, color='b')
plt.show()

#using seaborn

import seaborn as sns
import matplotlib.pyplot as plt

sns.lmplot(x="spends", y="income", data=df,ci=None)

# Set x-axis and y-axis ranges
plt.xlim(1500,5500)  # Set x-axis range from 1500,5500
plt.ylim(30000,75000)  # Set y-axis range from 30000,75000

# Set plot title and labels
plt.title("Linear Regression with Seaborn")
plt.xlabel("spends")
plt.ylabel("income")

# Show the plot
plt.show()