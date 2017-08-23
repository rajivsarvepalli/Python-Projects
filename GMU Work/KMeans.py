import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.preprocessing import scale
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
class KMeans:
    def __init__(self,n_cluster=8,total_iter =300):
        self.n_cluster = n_cluster
        self.total_iter = total_iter
    def fit(self,X):
        plt.scatter(X[:,0],X[:,1],c="black")
        plt.draw()
        plt.pause(5)
        plt.clf()
        self.centroids = self.kmeans_plus_plus(X,self.n_cluster)
        for i in range(0,self.total_iter):
                self.draw(X)
                self.recomputecentroids(X)
    def clusters(self,X):
        cluster_labels= []
        for x in X:
            dist = []
            for c in self.centroids:
                dist.append(distance.euclidean(x,c))
            cluster_labels.append(np.argmin(dist))
        return cluster_labels
    def recomputecentroids(self,X):
        cluster_labels = self.clusters(X)
        for n in range(0,self.n_cluster):
            avg = []
            for i in range(0,len(cluster_labels)):
                if n==cluster_labels[i]:
                    avg.append(X[i])
            self.centroids[n] = np.mean(np.vstack(avg),axis=0)
    def kmeans_plus_plus(self,X,K):#intialization
        intial_cs = [X[0]]
        for k in range(1, K):
            Dx = scipy.array([min([scipy.inner(z-x,z-x) for z in intial_cs]) for x in X])
            probs = Dx/Dx.sum()
            probcomparison = probs.cumsum()
            rand = scipy.rand()
            for index,probability in enumerate(probcomparison):
                if rand < probability:
                    i = index
                    break
            intial_cs.append(X[i])
        return np.array(intial_cs)
    def predict(self,Y):
        x =self.clusters(Y)
        return x
    def draw(self,X):
        pred = self.predict(X)
        colors=['b','g','c','m','y','k','w','tab:orange','xkcd:peach','xkcd:reddish purple','xkcd:avocado']
        colors = colors[0:self.n_cluster]
        for i in range(len(pred)):
            for n in range(self.n_cluster):
                if pred[i]== n:
                    plt.scatter(X[i,0],X[i,1],c=colors[n])
        plt.scatter(self.centroids[:,0],self.centroids[:,1],c='r')
        plt.draw()
        plt.pause(1)
        plt.clf()
#x = np.loadtxt("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/UCI_machine_learning_Data_Repository/wine_dataset.txt",delimiter=",")
x,y = make_blobs(n_samples=100,n_features=10,centers=3,cluster_std=10,center_box=(-50.0, 50.0))
kmeans = KMeans(n_cluster=3,total_iter=10)
kmeans.fit(PCA(n_components=2).fit_transform(x))
plt.close()
print("done")
