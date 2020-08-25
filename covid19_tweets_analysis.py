# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:05:55 2020

@author: Anurag
"""
#Importing the lybrary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
import re

data = pd.read_csv('covid19_tweets.csv')
#Data exploration
data.describe()
data.head()
data['user_location'].isnull().any()
data['user_location'].isnull().sum() / data.shape[0]

#Which hashtags have been used most frequently
data['hashtags'].describe()
data.hashtags.unique()
data['hashtags'].value_counts()[:10].plot(kind='pie',figsize= (15,18))

# calculating the frequency of the tweets 
x1 = data.user_location.value_counts()

#plotting the top 20 tweeting locations of the world
data['user_location'].value_counts()[:20].plot(kind='barh',figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)
data['user_location'].value_counts()[:20].plot(kind='pie', figsize= (15,15), subplots=True )

# plotting the top 20 source of the tweets
data['source'].value_counts()[:25].plot(kind='barh', figsize= (8,10) )

#ploting the time frame of the tweets
data['date'].value_counts()[:25].plot(kind='barh', figsize= (8,10) )
data['date'].value_counts()[:50]
data['date'].describe()






#Taking out the text from the dataset
tex = data.loc[:,('text', 'user_location')]
pd.crosstab(data.user_location)
#Sentiment analysis using Textblob
from textblob import TextBlob
import string
tex = [doc.lower() for doc in tex]
tex.len
#tex = re.sub(r'http\S+', '', tex)
#tex = re.sub(r'^https?:\/\/.*[\r\n]*', '', tex, flags=re.MULTILINE)
tex = re.sub(r'\[[0-9]*\]',' ', tex[ :,0:1 ])






