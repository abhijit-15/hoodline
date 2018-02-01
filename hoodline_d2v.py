#Implements doc2vec model for clustering
from __future__ import print_function
import gensim
import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from hoodline_utils import *
from sklearn.cluster import KMeans
from hoodline_cluster import *

pd.options.mode.chained_assignment = None

def preprocess(data):
	doc_list = []
	data["summary"] = ""
	for i in range(len(data)):
		data["summary"][i] = HTML_TO_TEXT(data["content"][i])
		doc_list.append(data["summary"][i])

	en_stop = nltk.corpus.stopwords.words('english')
	s_stemmer = nltk.stem.snowball.SnowballStemmer("english")	

	taggeddoc = []
	texts = []  
	for index,i in enumerate(doc_list):
		# for tagged doc
		wordslist = []
		tagslist = []

		# clean and tokenize document string
		raw = i.lower()
		tokens = tokenize_only(raw)

		# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in en_stop]

		# remove numbers
		number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
		number_tokens = ' '.join(number_tokens).split()

		# stem tokens
		stemmed_tokens = [s_stemmer.stem(i) for i in number_tokens]
		# remove empty
		length_tokens = [i for i in stemmed_tokens if len(i) > 1]
		# add tokens to list
		texts.append(length_tokens)
	
		k = ' '.join([x for x in stemmed_tokens])

		td = TaggedDocument(k.split(),[str(index)])
		taggeddoc.append(td)
	
	return taggeddoc


docLabeled = preprocess(pd.read_csv("hoodline_challenge.csv")) 
model = gensim.models.doc2vec.Doc2Vec(dm=0,size=100, window=10, min_count=0, workers=11, alpha=0.025, min_alpha=0.01) #Bag of Words (dm = 0). Skip n-gram (dm = 1)
model.build_vocab(docLabeled)

for epoch in range(5):
    model.train(docLabeled , total_examples=model.corpus_count , epochs=10)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
  
textVect = model.docvecs.doctag_syn0

model.save('d2v_hoodline_300_10.pkl')

data = pd.read_csv('hoodline_challenge.csv')

model = gensim.models.doc2vec.Doc2Vec.load('d2v_hoodline_300_10.pkl')
print("Loaded model successfully")

textVect = model.docvecs.doctag_syn0
print(textVect.shape)

## K-means ##
num_clusters = 10
km = KMeans(n_clusters=num_clusters)
km.fit(textVect)
clusters = km.labels_.tolist()

data["cluster"] = clusters

## Print Example Headlines for Clusters ##
print_headlines(data , num_clusters, num_headlines=6)
