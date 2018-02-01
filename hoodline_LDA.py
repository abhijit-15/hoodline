import os
import time
import nltk
import gensim
import html2text
import pandas as pd
from gensim import corpora, models
from hoodline_utils import *

os.chdir(os.getcwd())

pd.options.mode.chained_assignment = None
data = pd.read_csv("hoodline_challenge.csv")	

# create English stop words list
en_stop = nltk.corpus.stopwords.words('english')

# Create stemmer
p_stemmer = nltk.stem.snowball.SnowballStemmer("english")

doc_set = []
data["summary"] = ""    
# create sample documents
for i in range(len(data)):
	data["summary"][i] = HTML_TO_TEXT(data["content"][i])
	doc_set.append(data["summary"][i])

# list for tokenized documents in loop
texts = []

tic = time.time()

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenize_only(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

toc = time.time()
print("Time for processing... "),
print(toc - tic)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

tic = time.time() #Approximate runtime ~ 15 min

# generate LDA model

#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=10)
toc = time.time()
print("Time for LDA... ",)
print(toc - tic)

#ldamodel.save('lda_model_20_10.pkl')
#print(ldamodel.print_topics(num_topics=20, num_words=10))

ldamodel = gensim.models.ldamodel.LdaModel.load('lda_model_20_10.pkl')
ldamodel.print_topics(num_topics=20, num_words=10)
