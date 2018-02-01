#Contains all utility functions used for solving the hoodline dataset challenge
import re
import os
import nltk
import time
import mpld3
import codecs
import sklearn
import html2text
import numpy as np
import pandas as pd
from BeautifulSoup import BeautifulSoup


def HTML_TO_TEXT(html):
    soup = BeautifulSoup(html)
    text_parts = soup.findAll(text=True)
    text = ''.join(text_parts)
    return text

def processData(data):
    tdata = data
    h = html2text.HTML2Text()
    h.ignore_links = True
    tdata["summary"] , tdata["links"] , tdata["stemmed"] , tdata["tokens"] = "" , "" , "" , ""

    tic = time.time()
    for i in range(len(tdata)) :#range(5):
	tdata["summary"][i] = HTML_TO_TEXT(data["content"][i])                                  #Gets only the textual part of the content
	tdata["summary"][i] = tdata["summary"][i].lower()				        #Converts to lowercase for each description
	tdata["links"][i] = re.findall(r'(http(s)?://[^\s]+)', tdata["content"][i])             #Gets all links present in the content
	tdata["stemmed"][i] = tokenize_and_stem(tdata["summary"][i])                            #Tokenizes and Stems each article
	tdata["tokens"][i]  = tokenize_only(tdata["summary"][i])			        #Only Tokenizes each article
    toc = time.time()
    print(toc - tic)
    return tdata

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def create_vocab_frame(data):
    totalvocab_tokenized = []
    totalvocab_stemmed = []
    h = html2text.HTML2Text()

    for i in data["summary"]:
        allwords_stemmed = tokenize_and_stem(i.decode('utf-8')) #for each item in 'summary', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) 		#extend the 'totalvocab_stemmed' list
    
        allwords_tokenized = tokenize_only(i.decode('utf-8'))
        totalvocab_tokenized.extend(allwords_tokenized)
    return pd.DataFrame({'words': allwords_tokenized}, index = allwords_stemmed)

def KL(P,Q):
     #Calculate KL divergence of two vectors
     """ Epsilon is used here to avoid conditional code for
     checking that neither P nor Q is equal to 0. """
     epsilon = 0.00001

     P = P + epsilon
     Q = Q + epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence
