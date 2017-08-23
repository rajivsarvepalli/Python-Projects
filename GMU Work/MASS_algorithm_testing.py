import time
from sklearn.cross_validation import train_test_split
from quick_pfp2_file_reader import alldata
from shortcuts import quick_pfp2_file_reader
from quick_txt_reader import allData
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors 
from MASS_algorithm import findNN
from shortcuts import memory_usage_psutil
if __name__=="__main__":
    times=[]
    start_time = time.time()
    s1 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State1")
    s2 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State2")
    s3 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State3")
    s4 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State4")
    sT = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/StateTamper")
    times+=[time.time()-start_time]
    c = np.concatenate((s1,s2,s3,s4),axis=0)
    query = c[0,100:900]
    c = np.concatenate((c,sT),axis=0)
    matrixProfile =[]
    start_time = time.time()
    for i in range(0,10):
        matrixProfile.append(findNN(s1[i],query))
    for i in range(0,10):
        matrixProfile.append(findNN(s2[i],query))
    for i in range(0,10):
        matrixProfile.append(findNN(s3[i],query))
    for i in range(0,10):
        matrixProfile.append(findNN(s4[i],query))
    for i in range(0,10):
        matrixProfile.append(findNN(sT[i],query))
    times+=[time.time()-start_time]
    start_time = time.time()
    matrixProfile = np.array(matrixProfile)
    a = np.sum(matrixProfile,axis=1)
    times+=[time.time()-start_time]
    start_time = time.time()
    from shortcuts import bar_stack_grapher
    times+=[time.time()-start_time]
    print(times)
    bar_stack_grapher(times,['Times'],['b','g','y','r'],legend_values=['Read','Matrix P(50)','nparray','import'])
    print(a/10000)
