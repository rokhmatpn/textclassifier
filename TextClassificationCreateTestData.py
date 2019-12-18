# LOAD DATA #

import pymssql
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

conn = pymssql.connect("10.11.22.100", "sa", "sqlbi4dm!n", "TARKDW")
cursor = conn.cursor()
cursor.execute("select distinct(category) from gn4categories")

filename = 'dataset.txt'
filecategories = 'categories.txt'

if os.path.exists(filename):
  os.remove(filename)

if os.path.exists('dataset_title.txt'):
  os.remove("dataset_title.txt")

# get parent categories 
categories = []
for row in cursor:
	dirCat = row[0].encode("utf-8", 'replace').decode()
	categories.append(dirCat)

j = 1
for cat in categories:
	print(cat)
	catfile = open(filecategories, "a+")
	catfile.write('%s\t%s\n' % (j, cat))
	catfile.close();

	#get files for each category
	curStories = conn.cursor()
	curStories.execute("""
		select a.id, a.title, a.category, c.category, a.body 
		from dumpforpostagger a 
			join GN4Categories c on (a.category=c.subcategory) where c.category = '%s'""" % cat)
	i = 1
	for rowStories in curStories:
		textdata = rowStories[4].encode("ascii", 'replace').decode().replace("?","").replace("\n"," ").replace("\r"," ").replace("\t"," ").replace("\"","")
		textdata = stemmer.stem(textdata)

		print("creating story number:%s id:%s" % (i, filename))
		newfile = open(filename, "a+")
		newfile.write('%s\t%s\n' % (textdata, cat))
		newfile.close();

		#test data title only
		textdata = rowStories[1].encode("ascii", 'replace').decode().replace("?","").replace("\n"," ").replace("\r"," ").replace("\t"," ").replace("\"","")
		textdata = stemmer.stem(textdata)

		print("creating title story number:%s id:%s" % (i, "dataset_title.txt"))
		newfile = open("dataset_title.txt", "a+")
		newfile.write('%s\t%s\n' % (textdata, cat))
		newfile.close();


		print(i)

		i += 1		

	j += 1
#	break

conn.close()

