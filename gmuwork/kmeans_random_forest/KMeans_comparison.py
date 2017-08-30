from KMeans import KMeans
from sklearn.cluster import KMeans as km
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
reduced_data,check = make_blobs(n_samples = 1000,n_features=2,centers=3,cluster_std=7)
start_time = time.time()
kmeans = KMeans(n_cluster=3,total_iter =300)
kmeans.fit(reduced_data)
pred = kmeans.predict(reduced_data)
print(time.time()-start_time)
plt.figure(1)
for i in range(0,len(pred)):
    if pred[i] ==0:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "b",alpha=.5)
    if pred[i] ==1:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "g",alpha=.5)
    if pred[i] ==2:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "y",alpha=.5)
plt.scatter(kmeans.centroids[0,0],kmeans.centroids[0,1],marker="*",c="r")  #wrong kmenas good plots
plt.scatter(kmeans.centroids[1,0],kmeans.centroids[1,1],marker="*",c="r")
plt.scatter(kmeans.centroids[2,0],kmeans.centroids[2,1],marker = "*",c="r")
start_time  = time.time()
kmeans = km(n_clusters=3)
kmeans.fit(reduced_data)
pred = kmeans.predict(reduced_data)
print(time.time()-start_time)
plt.figure(2)
for i in range(0,len(pred)):
    if pred[i] ==0:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "b",alpha=.5)
    if pred[i] ==1:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "g",alpha=.5)
    if pred[i] ==2:
        plt.scatter(reduced_data[i,0],reduced_data[i,1],c = "y",alpha=.5)
plt.scatter(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1],marker="*",c="r")#wrong kmenas good plots
plt.scatter(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1],marker="*",c="r")
plt.scatter(kmeans.cluster_centers_[2,0],kmeans.cluster_centers_[2,1],marker = "*",c="r")
plt.show()