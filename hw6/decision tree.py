# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pprint import pprint
#make dataframe
df = pd.DataFrame({'District': ['Suburban', 'Suburban', 'Rural', 'Urban', 'Urban', 'Urban', 'Rural', 'Suburban', 'Suburban', 'Urban', 'Suburban', 'Rural', 'Rural', 'Urban'],
        'House Type': ['Detached', 'Detached', 'Detached', 'Semi-detached', 'Semi-detached', 'Semi-detached', 'Semi-detached', 'terrace', 'Semi-detached', 'terrace', 'terrace', 'terrace', 'Detached', 'terrace'],
        'PreviousCustomer': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'Income': ['High', 'High', 'High', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'High'],
        'Outcome': ['Not responded', 'Not responded', 'Responded', 'Responded', 'Responded', 'Not responded', 'Responded', 'Not responded', 'Responded', 'Responded', 'Responded', 'Responded', 'Responded', 'Not responded']}
)
print(df)
print()
#features
features = df[["District","House Type","PreviousCustomer","Income"]]
#target feature
target = "Outcome"

#entropy function
def Entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Information Gain Function
def InfoGain(data, attribute, target_col):
    # Calculate total entropy
    total_entropy = Entropy(data[target_col])

    # Calculate weighted entropy
    element, count = np.unique(data[attribute], return_counts=True)
    child_entropy = np.sum(
        [(count[i] / len(data[attribute])) * Entropy(
            data.where(data[attribute] == element[i]).dropna()[target_col])
         for i in range(len(element))])

    # Calculate information gain
    info_Gain = total_entropy - child_entropy
    # Return information gain
    return info_Gain

#Represented by the third decimal point
print("District's info Gain:", round(InfoGain(df, "District", "Outcome"), 3) )
print("House Type's info Gain:", round(InfoGain(df, "House Type", "Outcome"), 3) )
print("PreviousCustomer;s info Gain:",round(InfoGain(df,"PreviousCustomer","Outcome"),3) )
print("Income's info Gain:", round(InfoGain(df, "Income", "Outcome"), 3) )
print()
# decisionTree function
def decisionTree(data,originaldata,features,target_attribute_name,parent_node_class = None):
    #if target feature has a single values,  returns its destination property
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    #if there is no data, Returns the target property with the maximum value from the source data
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    #if there is no features,return parent node's target features
    elif len(features) ==0:
        return parent_node_class

    else:# Defining the target properties of the parent node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        #select features to divide datas
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        #The largest infogain value is best_feature
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}} #make tree structure

        # Exclude technical attributes that show maximum information gain
        features = [i for i in features if i != best_feature]

        # growing branch
        for value in np.unique(data[best_feature]):
            #data division and dropna()
            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = decisionTree(sub_data,data,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree

        return(tree)

#print result tree
tree = decisionTree(df, df, ["District","House Type","PreviousCustomer","Income"], "Outcome")
print()
pprint(tree)

#prediction function
def predict_result(tree, data):

    for attribute, subtree in tree.items():
        value = data[attribute]
        if value not in subtree:
            return max(subtree, key=subtree.get)
            # If the attribute value is not in the subtree, return the majority class of the subtree
        subtree = subtree[value]
        if isinstance(subtree, dict):
            # If the subtree is a dictionary, recursively traverse it
            return predict_result(subtree, data)
        else:
            # If the subtree is a leaf node, return the predicted outcome
            return subtree

#predict result about test data
test_data = {'District': 'Suburban', 'HouseType': 'Detached', 'Income': 'Low', 'PreviousCustomer': 'Yes'}
predicted_result = predict_result(tree, test_data)

print("test data's result :", predicted_result)