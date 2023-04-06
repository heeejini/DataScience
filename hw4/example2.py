import matplotlib.pyplot as plt
import pandas as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

usl = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#load dataset into pandas dataframe
df=pd.read_csv(usl,names=['sepal length','sepal width','petal length','petal width','target'])

#print(df)

features = ['sepal length','sepal width','petal length','petal width']

#print(df)

#seperate out the features
x=df.loc[:,features].values #features에 포함된 값들의 row를 모두 인덱싱 함
#'sepal length','sepal width','petal length','petal width'의 row값들 다 뽑아냄
#print(x)

#seperate out the target
y=df.loc[:,['target']].values

#standardize the features
x=StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2) #2개로 축소
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1','principal component 2'])

finalDf=pd.concat([principalDf, df[['target']]], axis=1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets=['Iris-setosa','Iris-versicolor', 'Iris-virginica']
color=['r','g','b']
for target, color in zip(targets, color):
    indicesToKeep = finalDf['target']==target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c= color, s=50)

ax.legend(targets)
ax.grid()
plt.show()