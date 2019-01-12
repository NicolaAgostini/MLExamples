

import numpy as np;

import urllib.request;
urllib.request.urlretrieve(url = "http://cs.joensuu.fi/sipu/datasets/s1.txt", 
                           filename = "Cluster.txt");  #importo il dataset


data_set = np.loadtxt(fname = "Cluster.txt")

data_set

import matplotlib.pyplot as plt;

plt.scatter(data_set[:,0],data_set[:,1],c='black',marker='.');  #disegno il dataset

from scipy.cluster.hierarchy import dendrogram, linkage;

linked = linkage(data_set, 'single'); #eseguo il clustering agglomerativo (bottom-up) con misurazione della distanza tra due cluster Single-Link cioè la distanza tra due cluster è la minima distanza tra due esempi appartenenti a cluster differenti

dendrogram(linked,  
            orientation='top',   
            distance_sort='descending',
            show_leaf_counts=True);      #genero il dendogramma

from sklearn.cluster import AgglomerativeClustering;

cluster = AgglomerativeClustering(n_clusters=15, affinity='euclidean', linkage='ward'); #eseguo l'algoritmo HAC con misurazione della distanza tra due cluster Ward cioè minimizzare la somma della distanza al quadrato degli elementi appartenenti ad un cluster con il centroide del cluster a cui appartengono

cluster.fit_predict(data_set);

plt.scatter(data_set[:,0],data_set[:,1], c=cluster.labels_, cmap='rainbow'); #disegno i cluster conn l'algoritmo HAC agglomerativo

from sklearn.cluster import KMeans;

km_ = KMeans(n_clusters=15, init='k-means++', n_init=10, max_iter=100, tol=1e-04); #inizializzo k-means

km_results = km_.fit_predict(data_set);

plt.scatter(data_set[:,0],data_set[:,1], c=km_results, cmap='rainbow');    #disegno i cluster individuati con l'algoritmo k-means e i relativi centroidi.
plt.scatter(km_.cluster_centers_[:,0], km_.cluster_centers_[:,1], s=250, marker='*', c='black' , label='centroidi');
plt.legend();
plt.grid();
plt.show();

