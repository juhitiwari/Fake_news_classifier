#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:13:51 2018

@author: slytherin
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
df=pd.read_csv('fake_or_real_news.csv')
#print(df.head())
y=df.label
X_train, X_test, y_train, y_test = train_test_split(df["text"],y,test_size=0.33,random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
#print(count_vectorizer.get_feature_names()[:10])

tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)
#print(tfidf_vectorizer.get_feature_names()[:10])
#print(tfidf_train.A[:5])

count_df=pd.DataFrame(count_train.A,columns=count_vectorizer.get_feature_names())
tfidf_df=pd.DataFrame(tfidf_train.A,columns=tfidf_vectorizer.get_feature_names())
print(count_df.head())
print(tfidf_df.head())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)
print(count_df.equals(tfidf_df))



