#load the dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


import pickle

lasttime = datetime.datetime.now()
print 'Creating model'

#filepath_dict = {'yelp':   'sentiment_analysis/yelp_labelled.txt',
#                 'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
#                 'imdb':   'sentiment_analysis/imdb_labelled.txt',
#				 'kompas':   'dataset.txt'}
#
#df_list = []
#for source, filepath in filepath_dict.items():
#    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
#    df['source'] = source  # Add another column filled with the source name
#    df_list.append(df)
#
#df = pd.concat(df_list)
#print(df.iloc[0])

df = pd.read_csv('dataset_title.txt', names=['sentence', 'label'], sep='\t')

#split train data & test data
sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000);

#regresion
vectorizer = CountVectorizer()
lr = LogisticRegression()
classifier= Pipeline([('vector', vectorizer), ('lr', lr)])
classifier.fit(sentences_train, y_train)
score = classifier.score(sentences_test, y_test)

#save to model
filename = 'finalized_title_model.sav'
joblib.dump(classifier, open(filename, 'wb'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

topredict = [stemmer.stem('Ditolak, Berbagai Upaya Hukum Jessica Kumala Wongso demi Bebas Jeratan Hukum')]

#load model
loaded_model = joblib.load(open(filename, 'rb'))
result = loaded_model.predict(topredict)

print("predicted class1 : %s" % result)
print("Accuracy : %s", score)
print("elapsed time : %s" % (datetime.datetime.now() - lasttime))

