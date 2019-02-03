#!/usr/bin/env python
# coding: utf-8

# ### Name: Giridhari Lal Gupta
# #### Roll Number : 2018201019
# ## Decision Tree
# ##### Decision Tree classifier to predict which valuable employees will leave next . This tree helps in reducing the number of senior employees leaving the company by predicting the next bunch
# #### Data set downloaded from : http://researchweb.iiit.ac.in/~murtuza.bohra/decision_Tree.zip

# # Code Section

# # Import-Libraries

# In[65]:


import sys
import math
import pandas as pd
import numpy as np
from numpy import *
import scipy.stats as stats

import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Tree-Structure

# In[66]:


class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    def display(self, level = 0):
        if self.children == {}: 
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    def predicts(self, root, catAtt):
        if self.children == {}: 
            return self.attribute
        if self.attribute in catAtt:
            value = root[self.attribute]
            if value in self.children:
                subtree = self.children[value]
                return subtree.predicts(root, catAtt)
        else:
            val = 0
            for val1 in self.children.keys():
                val = val1
            feature_name, comparison_operator, value, pos = val.split(" ")
            if root[feature_name] <= float(value):
                Question = "{} <= {} left".format(feature_name, value)
                subtree = self.children[Question]
            else:
                Question = "{} <= {} right".format(feature_name, value)
                subtree = self.children[Question]
            
            return subtree.predicts(root, catAtt)


# # Count-Label

# In[67]:


def CountTotalLabels(labelData):
    positive, negative = 0, 0
    for label in labelData:
        if label == 0:
            positive += 1
        else:
            negative += 1
    return positive, negative


# # Entropy

# In[68]:


def getEntropy(examples, target):
    positive, negative = CountTotalLabels(examples[target])
    logv = log(2)
    logv = logv * (positive + negative)
    Entropy = 0.0
    if not positive == 0:
        Entropy = positive * (-log(positive/(positive + negative)))/logv
    if not negative == 0:
        Entropy = Entropy + negative * (-log(negative/(positive + negative)))/logv
    return Entropy


# ### Potential split points
# #### For the numerical or contigous attribute , in order to find the best split point the following steps are done :
# ##### 1. sort the whole column
# ##### 2. find all unique values
# ##### 3. sort the values in non-decreasing order
# ##### 4. store ( a + b ) / 2 for every consecutive value known as potential split points
# ##### 5. for every stored value we check which split gives maximum information gain and the one which gives maximum is considered as best split point
# #### This funtion returns potential points which will be used the " best_decision_attribute " function to find best split

# In[69]:


def get_potential_splits(data):
    unique_values = np.unique(data)
    potential_splits = []
    unique_values.sort()
    for index in range(len(unique_values)):
        if index != 0:
            current_value = unique_values[index]
            previous_value = unique_values[index - 1]
            potential_split = (current_value + previous_value) / 2
            potential_splits.append(potential_split)
    
    return potential_splits


# #### This fucnction splits the data into two parts 
# ####      For Categorical data at value say A at index i ,Data whose value at index i is equal to A and data whose value at index i is not queal to A
# ####      For contigous data at value say A at index i,Data whose value  at  index i  is  less  than  or equal to A and data whose value at index is greater than A

# In[70]:


def split_data(data, split_column, split_value):
    
    split_column_values = data[split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above


# # Get-Best-Attribute
# #### This perform finds the simplest split worth among all potential split purposes and therefore returns that potential point which supplies most info gain

# In[71]:


def getBestAttribute(examples, target, attributes, catAtt):
    baseEntropy = getEntropy(examples, target)
    TotalLength = len(examples)
    informationGain = []
    
    for attribute in attributes:
        if attribute in catAtt:
            groupedData = examples.groupby(attribute)
            totalEntropy = 0
            for key,exampleSubset in groupedData:
                del exampleSubset[attribute]
                entropyOfSubset = getEntropy(exampleSubset,target)
                totalEntropy += (len(exampleSubset)/ TotalLength)*entropyOfSubset
            informationGain.append([attribute, baseEntropy-totalEntropy, key])
        else:
            potential_splits = get_potential_splits(examples[attribute])
            totalEntropy, val, overall_entropy = 0, 0, 999
            for value in potential_splits:
                data_below, data_above = split_data(examples, split_column=attribute, split_value=value)
                n = len(data_below) + len(data_above)
                totalEntropy =  ((len(data_below) / n) * getEntropy(data_below, target) 
                                  + (len(data_above) / n) * getEntropy(data_above, target))
                if totalEntropy <= overall_entropy:
                    overall_entropy = totalEntropy
                    val = value
            informationGain.append([attribute,baseEntropy-overall_entropy,val])

    bestAttribute = max(informationGain, key=lambda x: x[1])
    return bestAttribute[0], bestAttribute[2]


# # Classify

# In[72]:


def classify_data(examples, target, attributes):
    if len(attributes) == 0:
        item_counts = examples[target].value_counts()
        max_item = item_counts.idxmax()
        return max_item
    else:
        return False


# # Data-Purity

# In[73]:


def data_purity(examples, target):
    uniques = examples.apply(lambda x: x.nunique()).loc[target]
    if uniques == 1:
        return True
    else:
        return False


# # ID3 Algorithm

# In[74]:


def decisionTree(examples, target, attributes1, catAtt):
    attributes = attributes1[:] # Make it as Local
    if data_purity(examples, target):
        return TreeNode(examples[target].iloc[0])

    max_item = classify_data(examples, target, attributes)
    if len(attributes) == 0:
        return TreeNode(max_item)
    
    
    bestAttribute, bestValue = getBestAttribute(examples, target, attributes, catAtt)
    attributeRoot = TreeNode(bestAttribute)
    
    if bestAttribute in catAtt:
        attributes.remove(bestAttribute)
        groupedData = examples.groupby(bestAttribute)
        for key,exampleSubset in groupedData:
                if len(exampleSubset) == 0:
                    item_counts = exampleSubset[target].value_counts()
                    max_item = item_counts.idxmax()
                    attributeRoot.children[key] = TreeNode(max_item)
                else:
                    attributeRoot.children[key] = decisionTree(exampleSubset.drop([bestAttribute],axis=1), target, attributes, catAtt)
    else:
        data_below, data_above = split_data(examples, split_column=bestAttribute, split_value=bestValue)

        if len(data_below) == 0 or len(data_above) == 0:
            attributes.remove(bestAttribute)
            item_counts = examples[target].value_counts()
            max_item = item_counts.idxmax()
            return TreeNode(max_item) 
        
        key = "{} <= {} left".format(bestAttribute, bestValue)
    
        attributeRoot.children[key] = decisionTree(data_below, target, attributes, catAtt)
        key = "{} <= {} right".format(bestAttribute, bestValue)

        attributeRoot.children[key] = decisionTree(data_above, target, attributes, catAtt)

    return attributeRoot


# # Train-Test-Split

# In[75]:


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# # Evaluating-Tree

# In[76]:


def EvaluatingTree(tree, test, target, catAtt):
    correct = 0
    TP, FN, FP = 0,0,0
    for i in range(0, len(test)):
        if str(tree.predicts(test.loc[i], catAtt)) == str(test.loc[i,target]):
            correct += 1
            if(test.loc[i, target] == 0):
                FP += 1
            else:
                TP += 1
        else:
            if(test.loc[i, target] == 0):
                FN += 1
    print("\nThe accuracy is: ", correct/len(test))
    X = TP + FN
    Y = TP + FP
    Recall, Precision = 1, 1
    if X:
        Recall = TP/ X
    if Y:
        Precision = TP/ Y
    print("\nRecall is : ", Recall)
    print("\nPrecision is : ", Precision)
    if Precision or Recall:
        print("\nF1 Score is : ", (2 * (Recall * Precision))/ (Recall + Precision))


# # Prediction

# In[79]:


def prediction(tree, test, catAtt):
    correct = 0
    for i in range(0, len(test)):
        print(str(tree.predicts(test.loc[i], catAtt)))


# # Main-Function

# In[77]:


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    target = "left"

    train, test = train_test_split(df, test_size = 0.2)
    test = test.reset_index()

    attributes = train.columns.tolist()
    attributes.remove(target)
    
    catAtt = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
    tree = decisionTree(train, target, attributes, catAtt)
    
    tree.display()
    EvaluatingTree(tree, test, target, catAtt)


# In[78]:


EvaluatingTree(tree, test, target, catAtt)


# In[ ]:


prediction(tree, test, catAtt)

