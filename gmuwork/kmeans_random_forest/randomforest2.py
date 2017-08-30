import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
from sklearn.cross_validation import train_test_split
from quick_pfp1_file_reader import alldata
times=[]
start_time = time.time()
v00 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")
v01 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter01")
v10 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")
v11 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter01")
v20 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")
v21 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter01")
v30 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")
v31 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter01")
v40 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")
v41 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")
times+=[time.time()-start_time]
d = np.concatenate((v00,v01,v10,v11,v20,v21,v30,v31),axis=0)  #ratio 3 folders to one
np.random.shuffle(d)
part1 = d[0:29964]
part2 = d[29964:len(d)]
d = np.concatenate((part1,v40,v41),axis=0)
l =[0]*29964+[1]*9988 #(780)
data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.8,random_state=4) 
start_time = time.time()
forest = RandomForestClassifier(n_estimators=10)
forest.fit(data_train,labels_train)
times+=[time.time()-start_time]
pickle.dump(forest,open("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/models_classification.sav","wb"))
print(forest.predict(data_test))
print(accuracy_score(labels_test,forest.predict(data_test)))
print(times)

