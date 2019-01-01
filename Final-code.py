#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np   
import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re, string, timeit
from string import punctuation
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import statsmodels.api as sm
from scipy.interpolate import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels as s
from numpy.polynomial import polynomial
import scipy
import math
from sys import stdout
from sklearn.preprocessing import PolynomialFeatures
import statsmodels as s
from numpy.polynomial import polynomial
import scipy
from sklearn import linear_model

#preprocessing////////////////////////////////

#set data------------------------------------------------------------------
train_data = h5.File('C:\Train.h5','r')

dataset_train= pd.read_hdf('C:\Train.h5')

dataset_test= pd.read_csv('C:/utf-8of StudentTest.csv')

#nan change to 0------------------------------------------------------------
dataset_train['Favs'] = dataset_train['Favs'].replace(np.nan, 0)

dataset_train['RTs'] = dataset_train['RTs'].replace(np.nan, 0)

dataset_train['Listed'] = dataset_train['Listed'].replace(np.nan, 0)

dataset_train['Following'] = dataset_train['Following'].replace(np.nan, 0)

dataset_train['Followers'] = dataset_train['Followers'].replace(np.nan, 0)

dataset_train['likes'] = dataset_train['likes'].replace(np.nan, 0)

dataset_train['tweets'] = dataset_train['tweets'].replace(np.nan, 0)

dataset_train['reply'] = dataset_train['reply'].replace(np.nan, 0)

dataset_train['rank'] = dataset_train['rank'].replace(np.nan, 0)


#create new column
dataset_train['clear_TweetC']=dataset_train['Tweet content'].astype(str)

dataset_train['clear_TweetC'] = dataset_train['clear_TweetC'].apply(lambda x:''.join([i for i in x 
                                                  if i not in string.punctuation]))

#remove punctuation---------------------------------------------------------
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

dataset_train['clear_TweetC'] = dataset_train['clear_TweetC'].apply(remove_punctuation)



# change to nan 
#train= train.fillna({'Tweet content':''})


# change sentece to split words-dictionary
#results = set()
#dataset_train['clear_TweetC'].str.lower().str.split().apply(results.update)


#stop words-----------------------------------------------------------------
tokens= dataset_train['clear_TweetC']
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
dataset_train['clear_TweetC'] = tokens


#word count------------------------------------------------------------------
dataset_train['totalwords'] = dataset_train['clear_TweetC'].str.count(' ') + 1

#drop not important column
dataset_train = dataset_train.drop(['Tweet content' , 'User Name' , 'URLs' , 'clear_TweetC' , 'Tweet Id'] , axis=1)
dataset_train

#rescale----------------------------------------------------------------------
x = dataset_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
dataset_train[[ 'Favs' , 'RTs' , 'Followers' , 'Following' , 'Listed' , 'likes' , 'tweets' , 'reply' , 'rank' , 'totalwords']] = min_max_scaler.fit_transform(dataset_train[[ 'Favs' , 'RTs' , 'Followers' , 'Following' , 'Listed' , 'likes' , 'tweets' , 'reply' , 'rank' , 'totalwords']])

#split to train,validate and test----------------------------------------------
train, validate, test = np.split(dataset_train.sample(frac=1), [int(.6*len(dataset_train)), int(.8*len(dataset_train))])

#feature correlation-----------------------------------------------------------
print(train.corr())

#Plot correlation


# In[125]:


# Simple linear regression//////////////////////////

feature = test['tweets']
target = test['rank']
def slope_intercept(feature, target):
    x=np.array(feature)
    y=np.array(target)
    slope=( ((np.mean(x)*np.mean(y)) - np.mean(x*y)) / 
       ((np.mean(x)*np.mean(x)) - np.mean(x*x)) )
    slope=round(slope,2)
    intercept=(np.mean(y) - np.mean(x)*m)
    intercept=round(intercept,2)
    return slope, intercept

slope, intercept = slope_intercept(feature, target)
def get_regression_predictions(feature, intercept, slope):
    return intercept+slope*feature

def get_residual_sum_of_squares(feature, target, intercept, slope):
    y_hat=intercept+feature*slope
    return ( (target-y_hat)**2).sum()

get_residual_sum_of_squares(feature, target, intercept, slope)
# برای تمامی فیچرها، get_residual_sum_of_squares پس از محاسبه 
# tweets 
# کمترین خطا را داشت
# در نتیجه بهترین مدل برای حالت 
# linear regression 
#می باشد  

# Create rank column for dataset_test

x=np.array(test['tweets']).reshape(-1,1)
y=np.array(test['rank']).reshape(-1,1)
regr = linear_model.LinearRegression()
reg = regr.fit(x,y)
y_hat_2 =reg.predict(x)
a= y_hat_2
a= a.reshape(-1, len(a))
columns=['rank']
dataset_test['rank'] = pd.DataFrame(a.reshape(-1, len(a)),columns=columns)


# In[128]:


print(dataset_test['rank'])


# In[ ]:




