import numpy as np
from MASS_algorithm import findNN
import matplotlib.pyplot as plt
from shortcuts import quick_txt_reader
from scipy.spatial import distance
import time
from shortcuts import quick_txt_reader
from MASS_algorithm import STAMP
# def zeroOneNorm(x):
#    x= np.array(x)
#    y=np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isinf(x)))
#    indices = (np.argwhere(y==True))[:,0]
#    x = x-np.min(x[indices])
#    y=np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isinf(x)))
#    indices = np.argwhere(y==True)
#    x = x/np.max(x[indices])
#   return x
if __name__=="__main__":
    g = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeA")[0:2]
    f = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeM")[0:2]
    #ind = np.argpartition(a, -4)[-5:] 5 largest values indices
    start_time = time.time()
    P,I = STAMP(g[0],500)
    print(time.time()-start_time)
    truth =[]
    times =[]
    s4 = time.time()
    for index in range(0,len(g)):
        a = g[index]
        m = f[index]
        max_values=[]
        s2 = time.time()
        for i  in range(500,len(a),1):
            distanceprofile = findNN(a,a[i-500:i])
            ind = np.argpartition(distanceprofile, -5)[-5:] #5 max values
            max_values.append(distanceprofile[ind])
        times+=[time.time()-s2]
        sum_max1_inx = np.sum(np.array(max_values),axis=1)
        sum_max1_iny = np.sum(sum_max1_inx)
        max_values2=[]
        s3 = time.time()
        for i  in range(500,len(m),1):
            distanceprofile = findNN(m,m[i-500:i])
            ind = np.argpartition(distanceprofile, -5)[-5:]
            max_values2.append(distanceprofile[ind])
        sum_max2_inx = np.sum(np.array(max_values2),axis=1)
        sum_max2_iny = np.sum(sum_max2_inx)
        times+=[time.time()-s3]
        truth.append(sum_max1_iny>sum_max2_iny)
    t2 = np.mean(times)
    times=[]
    times+=[t2]
    times+=[time.time()-s4]
    print(times)
    print(truth)
    print(len(truth))
    sum_treu =0
    sum_false =0
    print('True: ',sum(truth))
    print('False: ',len(truth)-sum(truth))




