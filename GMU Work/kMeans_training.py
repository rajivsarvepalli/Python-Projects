import numpy as np
import os
from sklearn.cluster import KMeans as km
import pickle
import time
from sklearn.cross_validation import train_test_split
from scipy.fftpack import rfft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
        try :
            temp = infile.index(".")
            index = len(infile)-temp
            file_extension = infile[-index:]
            if file_extension==".txt":
                z = extract(directory +"/" + infile)
                txtfiles.append(z)  
        except ValueError:
            pass
    return np.array(txtfiles)
times =[]
start_time = time.time()
a = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeA")
b = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeB")
c = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeC")
d = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeD")
m =allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeM")
times+=[time.time()-start_time]
d = np.concatenate((a,b,c,d,m),axis=0) 
l =[0]*378+[1]*99       #labels for data [0] normal [1] malicious  (378)
start_time = time.time()
data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.8,random_state=42) 
times+=[time.time()-start_time]
start_time = time.time()
#data_train= rfft(data_train) #run rfft
times+=[time.time()-start_time]
start_time = time.time()
km = km(n_clusters=2,max_iter=300,n_init=10)
km.fit(data_train)
times+=[time.time()-start_time]
forest = RandomForestClassifier(n_estimators=10)
forest.fit(data_train,labels_train)
print(forest.predict(data_test))
print(accuracy_score(labels_test,km.predict(data_test)))
print(accuracy_score(labels_test,forest.predict(data_test)))
pickle.dump(km,open("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/model_Kmeans.sav","wb"))
print("Time to read: " + str(times[0])+ "\nTime to split data: " + str(times[1])+ "\nTime to run fft: " + str(times[2])+"\nTime to run kmeans: " +str(times[3]))