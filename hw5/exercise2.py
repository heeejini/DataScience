# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#using pandas and Seaborn
# create a DataFrame from the given data

data = {'Age':[30,40,50,60,40],
        'Income':[200,300,800,600,300],
        'Yrs worked':[10,20,20,20,20],
        'Vacation':[4,4,1,2,5]}
df = pd.DataFrame(data, columns=['Age','Income','Yrs worked','Vacation'])
print(df+"\n")

print("=====Sample Covariance=====")
# calculate sample covariance
sample_covariance = pd.DataFrame.cov(df)
print(sample_covariance)
#sample covariance 's heat map
sn.heatmap(sample_covariance, annot=True, fmt='g')
plt.show()

print()

print("=====Population Covariance=====")
# calculate population covariance
population_covariance =pd.DataFrame.cov(df, ddof=0)
print(population_covariance)
#population covariance 's heat map
sn.heatmap(population_covariance, annot=True, fmt='g')
plt.show()

import numpy as np
#using numpy
#input data
Age=[30,40,50,60,40]
Income=[200,300,800,600,300]
Yrs_worked=[10,20,20,20,20]
Vacation=[4,4,1,2,5]

data=np.array([Age,Income, Yrs_worked, Vacation])
print(data)
print()

print("=====Sample Covariance=====")
#sample covairance matrix (N-1)
covMatrix = np.cov(data, bias =False) # bias = True means 1/n , bias = False means 1/(n-1)
print(covMatrix)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

print("=====Population Covariance=====")
#population covariance matrix (N)
covMatrix = np.cov(data, bias =True)
print(covMatrix)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()