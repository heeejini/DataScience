import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#make dataframe
dataset = pd.DataFrame({
    'HEIGHT(cm)': [158,158,158,160,160,163,163,160,163,165,165,165,168,168,168,170,170,170],
    'WEIGHT(kg)': [58,59,63,59,60,60,61,64,64,61,62,65,62,63,66,63,64,68],
    'T_SHIRT_SIZE' : ['M','M','M','M','M','M','M','L','L','L','L','L','L','L','L','L','L','L']})

test_data =np.array([[161,61]], dtype=np.int64)

def scaling(dataset, test_data):
    #using MinMaxScaler
    x = dataset.iloc[:, 0].to_numpy() #height
    y = dataset.iloc[:, 1].to_numpy()#weight
    test_x = test_data[:,0]
    test_y = test_data[:,1]

    length = [len(x), len(x)+1]

    concat_data_x= np.concatenate([x,test_x])
    concat_data_y = np.concatenate([y,test_y])

    #Standardization, Learning & Testing
    scaler = MinMaxScaler()
    scaled_data_x = scaler.fit_transform(concat_data_x[:, np.newaxis])
    scaled_data_y = scaler.fit_transform(concat_data_y[:, np.newaxis])

    #print(scaled_data_x) #print scaled x data
    #print(scaled_data_y) #print scaled y data
    arr_x= np.split(scaled_data_x, length)
    arr_y=np.split(scaled_data_y, length)

    #print(arr_x)

    #Makes scaled data one-dimensional
    x=arr_x[0].reshape(-1)
    x_test=arr_x[1].reshape(-1)
    y=arr_y[0].reshape(-1)
    y_test=arr_y[1].reshape(-1)

    return x, x_test, y, y_test

def distance(x, y, test_x, test_y):
    #Euclidean formula for distance calculation
    dis = np.sqrt((x - test_x)**2 + (y - test_y)**2)
    return dis
def calResult(data):
    # count M, L
    M = 0
    L = 0

    for row in data.itertuples():
        if row.T_SHIRT_SIZE=="M": #if size is m
            M = M+1
        else : #if size is L
            L = L+1

    #predict size
    if M > L:
        size = 'M'
    else:
        size = 'L'
    return size

#KNN algorithm function
def KNN (x,y,test_x,test_y, k):
    dist = distance(x,y,test_x,test_y).reshape(-1)

    dist_df = pd.Series(data=dist, index=dataset.index)
    sort_result = dist_df.sort_values() #List small values first
    sort_result = sort_result.head(k) #Determine only the top k value

    print("K-nearest distance")
    print(sort_result , "\n")

    #Determine only the top k value
    k_data = dataset.loc[sort_result.index]
    print(k_data)

    result = calResult(k_data)
    return result


x, x_test, y, y_test = scaling(dataset, test_data)

print("-------------- K=3 --------------")
K=3 #Number of neighborhoods to judge
result =  KNN(x, y , x_test, y_test,K)
print("[Prediction Result : ", result ,"]")
print()
print("-------------- K=5 --------------")
K=5 #Number of neighborhoods to judge
result = KNN(x, y , x_test, y_test,K)
print("[Prediction Result : ", result ,"]")

