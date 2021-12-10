# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:42:32 2021

@author: wanti
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# DO NOT MODIFY THIS CELL
df1 = pd.read_csv("https://drive.google.com/uc?export=download&id=1thHDCwQK3GijytoSSZNekAsItN_FGHtm")
df1.info()

# data explore
df1.describe()

# correlation among the columns
df1.corr()


# visual your data
# age
plt.hist(df1['Age'], bins=50,color = "skyblue",ec='blue',range=(17,97),alpha =0.5)
plt.xlim(17,97)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
# add in vertial line
plt.vlines(x=59,  colors='red', ls=':',ymin=0, ymax=35,lw=2, label='Median =59')
plt.legend(loc='upper right')
plt.show()


# visual your data
# Income
plt.hist(df1['Income'], bins=50,color = "skyblue",ec='blue',range=(12000,142000),alpha =0.5)
plt.xlim(12000,142000)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
# add in vertial line
plt.vlines(x=75078,  colors='red', ls=':',ymin=0, ymax=35,lw=2, label='Median =75078')
plt.legend(loc='upper right')
plt.show()

# scatter age vs income
plt.plot(df1['Age'],df1['Income'],  'o', color='lightblue')
plt.title("Age vs. Income")
plt.xlabel("Age")
plt.ylabel("Income")








############################################################
# clustering trying out
df1.columns



# Standardization the Data
scaler = StandardScaler()
features = ['Age', 'Income', 'SpendingScore', 'Savings']
df1[features] = scaler.fit_transform(df1[features])
# df1 = df1.drop(labels=['km_cluster'],axis =1)

df1.describe().transpose()


###############
# method 1: Kmeans

# the elbow method
# ideal value of silhoustte = 1, worst possible value of silhousette = -1
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df1)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df1, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");


plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");





# Kmeans final model with k=5
k_means = KMeans(n_clusters=5, random_state=42)
k_means.fit(df1)

# results
df1['km_cluster']= k_means.labels_

# Let's look at the centers
k_means.cluster_centers_



# In the case of K-Means, the cluster centers *are* the feature means - that's how K-Means is defined! Sweet!
scaler.inverse_transform(k_means.cluster_centers_)
# inverse back the transform
df1[features] = scaler.inverse_transform(df1[features])



###############
# method 2: DBSCAN
# minpt at least 3, minpts = num freature*2

# Standardization the Data
scaler = StandardScaler()
features = ['Age', 'Income', 'SpendingScore', 'Savings']
df1[features] = scaler.fit_transform(df1[features])


db = DBSCAN(eps=0.2, min_samples=8)
db.fit(df1)

db.labels_


silhouette_score(df1, db.labels_)


# elbow method
silhouettes = {}

epss = np.arange(0.1, 0.9, 0.1)
minss = [3, 4, 5, 6, 7, 8, 9, 10]

ss = np.zeros((len(epss), len(minss)))

for i, eps in enumerate(epss):
    for j, mins in enumerate(minss):
        db = DBSCAN(eps=eps, min_samples=mins).fit(df1)
        if len(set(db.labels_)) == 1:
            ss[i, j] = -1
        else:
            ss[i, j] = silhouette_score(df1, db.labels_, metric='euclidean')
    

plt.figure();
#plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
for i in range(len(minss)):
    plt.plot(epss, ss[:, i], label="MinPts = {}".format(minss[i]));
#plt.plot(epss, ss[:, 1]);
plt.title('DBSCAN, Elbow Method')
plt.xlabel("Eps");
plt.ylabel("Silhouette");
plt.legend();
#plt.savefig('out/simple_dbscan_elbow');


###
# the best one happen when  minpts =5 and eps = 0.3
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(df1)

db.labels_





###############################################
# method 3:Hierarchical (Agglomerative)


agg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
agg.fit(df1)

agg.labels_
silhouette_score(df1, agg.labels_)



plt.figure();

plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], c=agg.labels_);
plt.title("Agglomerative");
plt.xlabel('Age');
plt.ylabel('Income');


plt.figure();

plt.scatter(df1.iloc[:, 2], df1.iloc[:, 3], c=agg.labels_);
plt.title("Agglomerative");
plt.xlabel('SpendingScore');
plt.ylabel('Savings');


# Dendograms

import scipy.cluster

aggl = scipy.cluster.hierarchy.linkage(df1, method='ward', metric='euclidean')

# Plot the dendogram
plt.figure(figsize=(16, 8));
plt.grid(False)
plt.title("Mall Dendogram");  
dend = scipy.cluster.hierarchy.dendrogram(aggl); 