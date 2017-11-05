import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
import os
import pickle
from sklearn.tree import export_graphviz
import time
from sklearn.cross_validation import train_test_split
from gmuwork.shortcuts import quick_pfp2_file_reader as alldata
times =[]
start_time = time.time()
s1 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State1")
s2 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State2")
s3 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State3")
s4 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State4")
sT = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/StateTamper")
times+=[time.time()-start_time]
d = np.concatenate((s1,s2,s3,s4),axis=0)  #ratio 3 folders to one
np.random.shuffle(d)
part1 = d[0:780]
part2 = d[780:len(d)]
d = np.concatenate((part1,sT),axis=0)
l =[0]*780+[1]*195 #(780)
data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.8,random_state=4) 
start_time = time.time()
forest = RandomForestClassifier(n_estimators=128)
forest.fit(data_train,labels_train)
times+=[time.time()-start_time]
pickle.dump(forest,open("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/models_classification.sav","wb"))
print(forest.predict(data_test))
print(accuracy_score(labels_test,forest.predict(data_test)))
print(times)
times =[]
start_time = time.time()
s1 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State1")
s2 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State2")
s3 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State3")
s4 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State4")
sT = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/StateTamper")
times+=[time.time()-start_time]
d = np.concatenate((s1,s2,s3,s4,sT),axis=0)
l =[0]*780+[1]*195 #(780)
data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.1,random_state=4) 
start_time = time.time()
forest= pickle.load(open("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/models_classification.sav","rb"))
times+=[time.time()-start_time]
print(forest.predict(data_test))
print(accuracy_score(labels_test,forest.predict(data_test)))
print(times)