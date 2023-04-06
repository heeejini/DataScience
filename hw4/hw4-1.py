import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('C:\\Users\\User\\PycharmProjects\\Data Science\\hw4\\housing.csv')
df.fillna(0, inplace=True)

#load dataset into pandas dataframe

#print(df)
#longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity
features = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']

#print(df)

#seperate out the features
x=df.loc[:,features].values #Indexes all rows of values contained in features

#seperate out the target
y=df.loc[:,['ocean_proximity']].values

#standardize the features
x=StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
#n_components=2 ,It means the number of components to leave behind.
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1','principal component 2'])

finalDf=pd.concat([principalDf, df[['ocean_proximity']]], axis=1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
#ISLAND, <1H OCEAN, NEAR OCEAN, NEAR BAY, INLAND
targets=['NEAR BAY','<1H OCEAN','INLAND', 'NEAR OCEAN', 'ISLAND']
color=['r','g','b','c','y'] #represent as 5 colors

for target, color in zip(targets, color):
    indicesToKeep = finalDf['ocean_proximity']==target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c= color, s=50)

ax.legend(targets)
ax.grid()
plt.show()