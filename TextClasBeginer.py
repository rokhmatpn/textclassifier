#load the dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import pickle

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

df = pd.read_csv('dataset.txt', names=['sentence', 'label'], sep='\t')

#split train data & test data
sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000);

pickle.dump(sentences_train, open('bow.sav', 'wb'))

#creat bag of words
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

#regresion
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

#save to model
filename = 'finalized_model.sav'
joblib.dump(classifier, open(filename, 'wb'))

topredict = ['HAJIN, KOMPAS.com - Sebuah rekaman memperlihatkan seorang komandan Negara Islam Irak dan Suriah ( ISIS) tewas di pertempuran setelah ditinggalkan anak buahnya. Dilansir Daily Mirror Kamis (20/12/2018), rekaman yang berasal dari GoPro memperlihatkan komandan ISIS itu menyiapkan senapannya. Si komandan ISIS itu masih terus menyerang sebelum dia tertembak dan tewas. Videonya ditemukan oleh Pasukan Pertahanan Suriah (SDF). Berdasarkan laporan Idlib Post, peristiwa itu terjadi di Deir Ezzor, kota terbesar yang terletak di timur Suriah. Dalam akun media sosialnya, SDF menjelaskan para anggota ISIS berada dalam kebingungan dan memilih untuk tidak patuh kepada pemimpin mereka dengan meninggalkannya. Milisi Kurdi itu, diwartakan The Independent, tengah berada dalam operasi untuk membebaskan Hajin yang berada dalam kendali ISIS. Kota yang berlokasi di tepi sungai Eufrat itu merupakan benteng terakhir ISIS di Suriah. SDF mengumumkan telah menguasai sebagian besar Hajin. Pembebasan Hajin bakal menjadi batu pijakan penting bagi SDF yang sudah menjadi sekutu negara Barat untuk memerangi ISIS dalam empat tahun terakhir.']

#load model
loaded_model = joblib.load(open(filename, 'rb'))
result1 = loaded_model.predict(vectorizer.transform(topredict))

result2 = classifier.predict(vectorizer.transform(topredict))

print("predicted class1:%s" % result1)
print("predicted class2:%s" % result2)
print("Accuracy:", score)
