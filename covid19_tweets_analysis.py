# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:05:55 2020

@author: Anurag
"""
#Importing the library
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
<<<<<<< HEAD

#Taking out the text from the dataset
tex = data.loc[:,('text')]




#using different dataset for sentiment analysis
data2 = pd.read_csv("Sentiment_data.csv")
nltk.download()


x = data2.iloc[:,2 ]
y = data2.iloc[:,1]

# x = [nltk.word_tokenize(e) for e in data2['text']] 

#spliting the dataset in train and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.3, random_state= 42)

#using the tfidf model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(x_train).toarray()
test_x_vectors = vectorizer.transform(x_test).toarray()

#Using the logistic regression model

from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()

clf_log.fit(train_x_vectors, y_train)

y_pred = clf_log.predict(test_x_vectors)
clf_log.score(test_x_vectors, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#using Naive byaes complementNB


from  sklearn.naive_bayes import ComplementNB
#creating the classifier
clf_compnb = ComplementNB()
y_pred2 = clf_compnb.fit(train_x_vectors, y_train).predict(test_x_vectors)
confusion_matrix(y_test,y_pred2)
clf_compnb.score(test_x_vectors, y_test)











=======
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
>>>>>>> 1d7c796a8139d4cea74fc123bacfea5c5bf33253






    
    