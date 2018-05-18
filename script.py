#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:13:51 2018

@author: slytherin
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
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

#count_df=pd.DataFrame(count_train.A,columns=count_vectorizer.get_feature_names())
#tfidf_df=pd.DataFrame(tfidf_train.A,columns=tfidf_vectorizer.get_feature_names())
#print(count_df.head())
#print(tfidf_df.head())
#difference = set(count_df.columns) - set(tfidf_df.columns)
#print(difference)
#print(count_df.equals(tfidf_df))

nb_classifier=MultinomialNB()
nb_classifier.fit(count_train,y_train)
pred=nb_classifier.predict(count_test)
score=metrics.accuracy_score(y_test,pred)
#print(score)
cm=metrics.confusion_matrix(y_test,pred,labels=['FAKE','REAL'])
#print(cm)

nb_classifier.fit(tfidf_train,y_train)
pred=nb_classifier.predict(tfidf_test)
score=metrics.accuracy_score(y_test,pred)
#print(score)
cm=metrics.confusion_matrix(y_test,pred,labels=['FAKE','REAL'])
#print(cm)

alphas = np.arange(0,1,0.1)
def train_and_predict(alpha):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidf_train,y_train)
    pred=nb_classifier.predict(tfidf_test)
    score=metrics.accuracy_score(y_test,pred)
    return score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


class_labels = nb_classifier.classes_
feature_names = tfidf_vectorizer.get_feature_names()
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))
print(class_labels[0], feat_with_weights[:20])
print(class_labels[1], feat_with_weights[-20:])
