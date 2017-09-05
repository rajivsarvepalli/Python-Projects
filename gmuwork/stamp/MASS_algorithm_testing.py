import numpy as np
import matplotlib.pyplot as plt
from gmuwork.shortcuts import quick_txt_reader
from scipy.spatial import distance
import time
from gmuwork.shortcuts import quick_txt_reader
from MASS_algorithm_development import STAMP
from multiprocessing import Pool,freeze_support
from itertools import repeat
import matplotlib.pyplot as plt
def fft_time(X):
    from scipy.fftpack import fft,ifft
    X=np.fft.fft(X)
    start = time.time()
    np.fft.ifft(X) 
    print(time.time()-start)
if __name__=="__main__":
    import scipy.io as sio
    from gmuwork.shortcuts import quick_txt_reader
    matfile = sio.loadmat("C:/Users/Rajiv Sarvepalli/Downloads/testData.mat")
    data = matfile['data'][0]
    # for x in range(0,142):
    #     data+=[2,2,2,2,2,5,5,5,5,5,6,6,6,6,6,6,8,8,8,8,8,9,9,9,9,9,8,8,8,8,8,6,6,6,6,6,5,5,5,5,5,2,2,2,2,2]
    # data = np.insert(data,2500,[10,11,12,13,14,15,16,17,
    # 18,19,20,21,22,23,24,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10])
    # data = np.insert(data,1000,[10,10,10,10,10])
    # data = np.insert(data,4500,[10,11,11,11,11,11,11,11,11,11,10])
    AtoA = np.load("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/AtoA.txt.npy")
    AtoM = np.load("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/AtoM.txt.npy")
    print(np.max(AtoA))
    print('loc: ',np.argmax(AtoA))
    print(np.max(AtoM))
    print('loc: ',np.argmax(AtoM))
    testDataStamp = np.load("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/testDataStamp.npy")
    testdataMAtrixprolfile = testDataStamp[0]
    times = [x for x in range(0,len(data))]
    plt.figure(figsize=(30,5))
    plt.plot(times,data,c='r')
    times = [x for x in range(0,len(testdataMAtrixprolfile))]
    plt.figure(figsize=(30,5))
    plt.plot(times,testdataMAtrixprolfile,c='b')
    plt.show()
    
