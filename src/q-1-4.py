#!/usr/bin/env python
# coding: utf-8

# ### Name: Giridhari Lal Gupta
# #### Roll Number : 2018201019
# ## Decision Tree
# ##### Decision Tree classifier to predict which valuable employees will leave next . This tree helps in reducing the number of senior employees leaving the company by predicting the next bunch
# #### Data set downloaded from : http://researchweb.iiit.ac.in/~murtuza.bohra/decision_Tree.zip

# # Code Section

# # Import-Libraries

# In[123]:


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


# # Entropy

# In[124]:


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

# In[125]:


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
                totalEntropy += (len(exampleSubset)/ TotalLength) * entropyOfSubset
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


# ## Part 4 : Visualise training data on a 2-dimensional plot

# In[126]:


def decisionTree(examples, target, attributes, catAtt):
    att = []
    for _ in range(0,2):
        bestAttribute, bestValue = getBestAttribute(examples, target, attributes, catAtt)
        attributeRoot = TreeNode(bestAttribute)
        attributes.remove(bestAttribute)
        att.append(bestAttribute)
    return att


# # Train-Test-Split

# In[127]:


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# # Main-Function

# In[147]:


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    
    target = "left"

    train, test = train_test_split(df, test_size = 0.2)
    test = test.reset_index()
    
    attributes = train.columns.tolist()
    attributes.remove(target)
    
    catAtt = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']

    bestAttribute = getFirstTwoBestAttribute(train, target, attributes, catAtt)


# ### Graph between two having maximum information gain

# In[155]:


sns.catplot(x = bestAttribute[0], y = bestAttribute[1], hue="left", data=train)


# In[ ]:




