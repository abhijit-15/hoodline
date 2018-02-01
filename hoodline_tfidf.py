from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hoodline_utils import tokenize_and_stem

#define vectorizer parameters
"""
max_df: this is the maximum frequency within the documents a given feature can have to be used in the tfi-idf matrix.
	If the term is in greater than 85% of the documents it probably cares little meanining.

min_idf: this could be an integer (e.g. 5) and the term would have to be in at least 5 of the documents to be considered.
	 Here I pass 0.2; the term must be in at least 20% of the document. Lower values would mean that.

ngram_range: Use unigrams, bigrams and trigrams.
"""

def generate_tfidf(data):

	tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

	tfidf_matrix = tfidf_vectorizer.fit_transform(data) #fit the vectorizer to tokens

	terms = tfidf_vectorizer.get_feature_names()

	dist = 1 - cosine_similarity(tfidf_matrix)

	return tfidf_matrix , terms , dist
