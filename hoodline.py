#Solve hoodline challenge

#Kindly uncomment relevant sections

#Imports
import os
from hoodline_utils import * #Helper Functions
from hoodline_tfidf import *
from hoodline_cluster import *

os.chdir(os.getcwd())

pd.options.mode.chained_assignment = None

if __name__ == "__main__":

	#data = pd.read_csv("hoodline_challenge.csv") 						#Read Original Dataset
	#k = processData(data)									#Process Dataset (Handle urls, stemming, tokenization)
	#k.to_csv("hoodline_dataset_processed.csv",header=True, index=False, encoding='utf-8')	#Save Processed Dataset

	data = pd.read_csv("hoodline_dataset_processed.csv") 					#Load processed dataset
	
        vocab_frame = create_vocab_frame(data)							#Vocabulary of all tokenized and stemmed words (Lookup table)

	stopwords = nltk.corpus.stopwords.words('english')
	stemmer = nltk.stem.snowball.SnowballStemmer("english")

	#tf-idf k-means------------------------------------------------------------------------------------------------------------------------------------------
	#Approximate Run Time ~ 5 minutes	
	tfidf_matrix , terms , dist = generate_tfidf(data["summary"])
	#print(terms)										#List of the features used in the tf-idf matrix. (Vocabulary)

	data["cluster"] = ""
	num_clusters = 10
	data["cluster"] = do_kmeans(tfidf_matrix , num_clusters) 				#Performs k-means clustering with 20 clusters
	print(data["cluster"].unique())
	#print(data["cluster"].value_counts().sort_index())					#See how many articles are assigned to which cluster
	km = joblib.load('tfidf_kmeans_doc_cluster.pkl') 					#Loads the converged k-means model	
	#print_cluster_terms(km , num_clusters , terms , vocab_frame , data) 			#Tells what are the important keywords for each cluster
	#print_headlines(data , num_clusters, num_headlines=6)					#Prints headlines tagged to each cluster
	#--------------------------------------------------------------------------------------------------------------------------------------------------------

	#doc2-vec k-means and LDA is implemented in separate files called hoodline_d2v.py and hoodline_LDA.py respectively
	

