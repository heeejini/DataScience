import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

A=[45,37,42,35,39]
B=[38,31,26,28,33]
C=[10,15,17,21,12]

data=np.array([A,B,C])
#population covariance matrix N
covMatrix = np.cov(data, bias=True)
print(covMatrix)
sn.heatmap(covMatrix,annot=True,fmt='g')
plt.show()