import numpy as np  
import re  
from sklearn.datasets import load_files  
import pickle  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier

#load trainingdata
article_data = load_files('KompasTrainingDoc2016')  
X, y = article_data.data, article_data.target  

#create bag of word
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)  
X = vectorizer.fit_transform(X).toarray()  

#convert to tfidf
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()  
 
#split train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#train
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)  

#predict
y_pred = classifier.predict(X_test)  

print y
print y_pred