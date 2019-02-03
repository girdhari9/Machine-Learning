#!/usr/bin/env python
# coding: utf-8

# ### Name: Giridhari Lal Gupta
# #### Roll Number : 2018201019
# ## Decision Tree
# ##### Decision Tree classifier to predict which valuable employees will leave next . This tree helps in reducing the number of senior employees leaving the company by predicting the next bunch
# #### Data set downloaded from : http://researchweb.iiit.ac.in/~murtuza.bohra/decision_Tree.zip

# # Code Section

# ## Import-Libraries

# In[25]:


import sys
import pandas as pd
import numpy as np
from numpy import *
import scipy.stats as stats
import random


# # Tree-Structure

# In[28]:


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
     
    def predicts(self, root):
        if self.children == {}: 
            return self.attribute
        value = root[self.attribute]
        if value in self.children:
            subtree = self.children[value]
            return subtree.predicts(root)


# # Count-Label
# ### Count number of zeros and ones in label column Data

# In[14]:


def CountTotalLabels(labelData):
    positive, negative = 0, 0
    for label in labelData:
        if label == 0:
            positive += 1
        else:
            negative += 1
    return positive, negative


# # Entropy

# In[15]:


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


# # Get-Best-Attribute
# #### This perform finds the simplest split worth among all potential split purposes and therefore returns that potential point which supplies most info gain

# In[16]:


def getBestAttribute(examples, target, attributes):
    baseEntropy = getEntropy(examples, target)
    TotalLength = len(examples)
    informationGain = []
    
    for attribute in attributes:
        groupedData = examples.groupby(attribute)
        totalEntropy = 0
        for key,exampleSubset in groupedData:
            del exampleSubset[attribute]
            entropyOfSubset = getEntropy(exampleSubset,target)
            totalEntropy += (len(exampleSubset)/ TotalLength)*entropyOfSubset
        informationGain.append([attribute, baseEntropy-totalEntropy])

    bestAttribute = max(informationGain, key=lambda x: x[1])
    return bestAttribute[0]


# # Classify
# ##### This operate finds the info price that represents the leaf throughout prediction time, if there's still impurity within the leaf node and hench are going to be having completely different label values thus we decide the worth at the time of prediction that happens most of the time within the node.

# In[17]:


def classify_data(examples, target, attributes):
    if len(attributes) == 0:
        item_counts = examples[target].value_counts()
        max_item = item_counts.idxmax()
        return max_item
    else:
        return False


# # Data-Purity?
# ##### If all the values at "Left" column square measure same then that's thought of as pure node and isn't any splitted .Hence thought of as leaf node

# In[18]:


def data_purity(examples, target):
    uniques = examples.apply(lambda x: x.nunique()).loc[target]
    if uniques == 1:
        return True
    else:
        return False


# # ID3 Algorithm
# ### Decision Tree Training Model
# #### This function takes the data ,trains the model and returns the decision tree

# In[19]:


def decisionTree(examples, target, attributes1):
    attributes = attributes1[:] # Make it as Local
    if data_purity(examples, target):
        return TreeNode(examples[target].iloc[0])

    max_item = classify_data(examples, target, attributes)
    if len(attributes) == 0:
        return TreeNode(max_item)
    
    bestAttribute = getBestAttribute(examples, target, attributes)
    attributeRoot = TreeNode(bestAttribute)
    attributes.remove(bestAttribute)
    groupedData = examples.groupby(bestAttribute)
    
    for key,exampleSubset in groupedData:
            if len(exampleSubset) == 0:
                item_counts = exampleSubset[target].value_counts()
                max_item = item_counts.idxmax()
                attributeRoot.children[key] = TreeNode(max_item)
            else:
                attributeRoot.children[key] = decisionTree(exampleSubset.drop([bestAttribute],axis=1), target, attributes)
    return attributeRoot


# # Train-Test-Split

# In[20]:


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# # Evaluating-Tree
# ### Calculation Report
# #### This function calculates TP, TN, FP, FN ,Accuracy ,Precision, Recall, F1-score
# ##### Accuracy  =  ( TP + TN ) / ( TP + FP + TN + FN )
# ##### Precsion  = TP / ( TP + FP )
# ##### Recall  = TP / ( TP + FN )
# ##### F1 = (2  x Precision x) / ( Precision + Recall) 

# In[21]:


def EvaluatingTree(tree, test, target):
    correct = 0
    TP, FN, FP = 0,0,0
    for i in range(0, len(test)):
        if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
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


# # Main-Function

# In[30]:


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    target = "left"
    raw_att = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
    
    train, test = train_test_split(df, test_size = 0.2)
    test = test.reset_index()
    
    test = test.drop(raw_att, axis = 1)
    test = test.drop(['index'], axis = 1)
    train = train.drop(raw_att, axis = 1)
    
    attributes = train.columns.tolist()
    attributes.remove(target)
    
    tree = decisionTree(train, target, attributes)
    tree.display()


# In[31]:


EvaluatingTree(tree, test, target)


# In[ ]:




