#load the dataset
import pandas as pd
from sklearn.externals import joblib
import pickle
import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import sys

if len(sys.argv) == 1:
	print '''
Need argument
python TextClassifierFromModel.py "topredict" "model"'''
	exit()

lasttime = datetime.datetime.now()

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

if len(sys.argv) == 3 :
	filename = sys.argv[2]
else:
	filename = 'finalized_model.sav'
#topredict = [stemmer.stem('HAJIN, KOMPAS.com - Sebuah rekaman memperlihatkan seorang komandan Negara Islam Irak dan Suriah ( ISIS) tewas di pertempuran setelah ditinggalkan anak buahnya. Dilansir Daily Mirror Kamis (20/12/2018), rekaman yang berasal dari GoPro memperlihatkan komandan ISIS itu menyiapkan senapannya. Si komandan ISIS itu masih terus menyerang sebelum dia tertembak dan tewas. Videonya ditemukan oleh Pasukan Pertahanan Suriah (SDF). Berdasarkan laporan Idlib Post, peristiwa itu terjadi di Deir Ezzor, kota terbesar yang terletak di timur Suriah. Dalam akun media sosialnya, SDF menjelaskan para anggota ISIS berada dalam kebingungan dan memilih untuk tidak patuh kepada pemimpin mereka dengan meninggalkannya. Milisi Kurdi itu, diwartakan The Independent, tengah berada dalam operasi untuk membebaskan Hajin yang berada dalam kendali ISIS. Kota yang berlokasi di tepi sungai Eufrat itu merupakan benteng terakhir ISIS di Suriah. SDF mengumumkan telah menguasai sebagian besar Hajin. Pembebasan Hajin bakal menjadi batu pijakan penting bagi SDF yang sudah menjadi sekutu negara Barat untuk memerangi ISIS dalam empat tahun terakhir.')]
topredict = [stemmer.stem(sys.argv[1])]

#load model
loaded_model = joblib.load(open(filename, 'rb'))
result = loaded_model.predict(topredict)

elapsedtime = datetime.datetime.now() - lasttime

print("predicted class : %s" % result)
print("elapsed time : %s" % elapsedtime)
