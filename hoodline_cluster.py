#Use for clustering data
from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def do_kmeans(tfidf_matrix , num_clusters): 

	km = KMeans(n_clusters=num_clusters)

	km.fit(tfidf_matrix)

	clusters = km.labels_.tolist()	

	#joblib.dump(km,'tfidf_kmeans_doc_cluster.pkl') #Saves K-means model

	km = joblib.load('tfidf_kmeans_doc_cluster.pkl') #Loads presaved K-means model
	
	clusters = km.labels_.tolist()

	return clusters


def print_cluster_terms(km , num_clusters , terms , vocab_frame , frame):	

	print("Top terms per cluster:")

	#sort cluster centers by proximity to centroid
	order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

	for i in range(num_clusters):
    		print("Cluster %d words:" % i, end='')
    
		for ind in order_centroids[i, :10]:
        		print('%s' % terms[ind]),


def print_headlines(data , num_clusters, num_headlines):
	for n in range(num_clusters):
		print("---------------------------------------------------------------------------------------------------------")
		counter = 0
		print("Cluster %d headlines: " % n)
		for i in range(len(data)):
			if data["cluster"][i] == n and counter < num_headlines:
				print(data["headline"][i])
				print("\n")
				counter += 1 

