import numpy as np
import matplotlib.pyplot as plt
import time
from MASS_algorithm_development import STAMP
from gmuwork.shortcuts import quick_pfp2_file_reader
from gmuwork.shortcuts import quick_pfp1_file_reader
from gmuwork.shortcuts import simple_line_graph_with_points
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
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:2]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:2]
    matrixProfileAtoA =[]
    matrixPRofileAtoM =[]
    start_time = time.time()
    for i in range(0,len(v00)-1):
        matrixProfileAtoA+=[STAMP(v00[i],512,None)]
        matrixPRofileAtoM+=[STAMP(v00[i],512,v40[i])]
    print(time.time()-start_time)
    print(np.amax(matrixProfileAtoA,axis=1))
    print(np.amax(matrixPRofileAtoM,axis=1))
    s1 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State1")[0]
    simple_line_graph_with_points([10000, 20000, 30000, 40000,60000],[72.85942721366882, 263.5356285572052, 637.4917480945587, 1170.4225871562958, 2521.917801141739],annotated_values=(True,1))
    times =[]
    n= []
    for i in range(20000,len(s1)//4,10000):
        start_time = time.time()
        STAMP(s1[0:i],256,None) 
        n+=[i]
        times+=[time.time()-start_time]
    for i in range(len(s1)//4,len(s1)//2,20000):
        start_time = time.time()
        STAMP(s1[0:i],256,None)
        n+=[i]
        times+=[time.time()-start_time]
    for i in range(len(s1)//2,len(s1),50000):
        start_time = time.time()
        STAMP(s1[0:i],256,None)
        n+=[i]
        times+=[time.time()-start_time]
    print(n)
    print(times)
    simple_line_graph_with_points(n,times,annotated_values=(True,1))







