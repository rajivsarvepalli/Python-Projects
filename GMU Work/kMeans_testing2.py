import pickle
#from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import rfft
def extract(file_name):
    file_object = open(file_name,"r")
    lin =file_object.read()
    line = lin.replace("\n","\t")#add
    line = line.replace(" ", "")
    z = [x for x in line.split("\t") if x]

    data = np.array(z,dtype=float)
    file_object.close()
    return data
def allData(directory):
    txtfiles=[]
    directories = os.listdir(directory)
    for infile in directories:
        temp = infile.find(".")
        if temp !=-1:
            index = len(infile)-temp
            file_extension = infile[-index:]
            if file_extension==".txt":
                z = extract(directory +"/" + infile)
                txtfiles.append(z)  
    return np.array(txtfiles)
def predict(X):
    mylist=[]
    for i in range(0,len(X)):
        if np.mean(X[i])>2.105:
            mylist+=[1]
        else:
            mylist+=[0]
    return mylist
a = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeA")
b = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeB")
c = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeC")
d = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeD")
m =allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeM")

d = np.concatenate((a,b,c,d,m),axis=0)
d = rfft(d)
l =[0]*378+[1]*99
km = pickle.load(open("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/model_Kmeans.sav","rb"))
y =km.predict(d)
print(accuracy_score(l,y))