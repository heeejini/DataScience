import numpy as np
import pandas as pd

df = pd.DataFrame({'column1':[3.,'?',2.,5.],
         'column2':["*",4.,5.,6.],
         'column3':["+",3.,2.,"&"],
        'column4':[5.,"?",7.,"!"]})

print(df)
missing_values=["?","*","&","!","!"]

#값 바꾸기 , inplace=True 해줘야 바뀜
df.replace({"?":np.nan ,"*":np.nan,"&":np.nan,"!":np.nan,"+":np.nan}, inplace=True)
print(df)

print(df.isna().sum()) #df.isna().sum() 행렬 별 NaN 갯수
print(df.isna().any()) #df.isna().any() 적어도 하나의 true가 있으면 true

print("---------dropna/axis=0---------") #가로
print(df.dropna(axis=0, how='all'))
print(df.dropna(axis=0, how='any'))
print("---------dropna/axis=1---------") #세로
print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, how='any'))
print()


print("-----fillna------")
#fillna with 100, mean, median
print(df.fillna(100))
mean = df['column1'].mean()
print(df['column1'].fillna(mean))
median = df['column2'].median()
print(df['column2'].fillna(median))

#ffill, bfill
# axis = 0 이라면,'가로에 대하여' 라고 이해하기
print(df.fillna(axis=0, method='ffill'))
print(df.fillna(axis=0, method='bfill'))
print(df.fillna(axis=1, method='ffill'))
print(df.fillna(axis=1, method='bfill'))
print()