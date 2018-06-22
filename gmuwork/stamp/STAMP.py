import numpy as np
from scipy.fftpack import fft,ifft #scipy fft is faster than numpy
import pyfftw
def MASSPRE(x,m):
    '''
    preparation for mass
    '''
    n =len(x)
    #x = np.append(x,(n)*[0])
    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(np.power(x,2))
    sumx2 = cum_sumx2[m:n] -cum_sumx2[0:n-m]
    sumx = cum_sumx[m:n]-cum_sumx[0:n-m]
    meanx = np.divide(sumx,m)
    sigmax2 = (sumx2/m)- np.power(meanx,2)
    sigmax = np.sqrt(sigmax2)
    X =fft(x)
    value = (sumx2 - 2*sumx*meanx + m*(np.power(meanx,2)))/sigmax2
    return n,cum_sumx,cum_sumx2,sumx2,sumx,meanx,sigmax2,sigmax,X,value
def MASS(y,n,cum_sumx,cum_sumx2,sumx2,sumx,meanx,sigmax2,sigmax,X,value):
    '''
    Runtime: O(n^2logn)
    Implementation of the MASS algorithm
    input: a time_series x, and query y
    output: a distance Profile computed \n
    by comparing every subsquence of the same length as y\n
    and computing their euclidean distance 
    '''
    #normlaize line below
    y = (y-np.mean(y))/np.std(y)
    m = len(y)
    y = y[::-1]
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y,2))
    y = np.append(y,((n)-(m))*[0])#changed from n-m to 
    Y = fft(y)
    Z = X*Y
    Z = pyfftw.byte_align(Z)
    z = pyfftw.interfaces.scipy_fftpack.ifft(Z)
    dist = value - 2*(z[m:n] - sumy*meanx)/sigmax + sumy2
    return np.abs(np.lib.scimath.sqrt(dist))#complex sqrt required
def STAMP(time_seriesA,m,time_seriesB):
    '''
    Runtime: O(n^2logn)
    input: a time_seriesA and a subLen for the sliding windows,\n
    a time_seriesB to compare time_seriesA, \n
    or None to self-join time_SeriesA to compare it to itself
    output: the matrix profile,followed by the integere pointers\n
    these pointers allow you to find nearest neighbor in O(n) time
    '''
    time_seriesA = np.array(time_seriesA)
    if len(np.shape(time_seriesA))!=1:
        time_seriesA = time_seriesA.flatten()
    if time_seriesB is None:
        time_seriesB = time_seriesA
    nB = len(time_seriesB)
    P = np.array((nB-m)*[np.inf])
    I = np.array((nB-m)*[0])
    n,cum_sumx,cum_sumx2,sumx2,sumx,meanx,sigmax2,sigmax,X,value = MASSPRE(time_seriesA,m)
    pyfftw.interfaces.cache.enable()
    for idx in range(0,(nB-m)):
        D = MASS(time_seriesB[idx:idx+m],n,cum_sumx,cum_sumx2,sumx2,sumx,meanx,sigmax2,sigmax,X,value)
        excludezones = max(1,idx-(m//2))
        excludezoneend = min(len(time_seriesA)-m,idx+(m//2))
        D[excludezones-1:excludezoneend] = np.inf
        P[D<=P] = D[D<=P]
        I[D<=P] = idx
    return P,I
if __name__ =="__main__":
    #testing
    import time
    import scipy.io as sio
    matfile = sio.loadmat("C:/Users/Rajiv Sarvepalli/Downloads/MP_first_test_penguin_sample.mat")
    data = matfile['penguin_sample'][0:20000]
    # for x in range(0,142):
    #     data+=[2,2,2,2,2,5,5,5,5,5,6,6,6,6,6,6,8,8,8,8,8,9,9,9,9,9,8,8,8,8,8,6,6,6,6,6,5,5,5,5,5,2,2,2,2,2]
    # data = np.insert(data,2500,[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10])
    # data = np.insert(data,1000,[10,10,10,10,10])
    # data = np.insert(data,4500,[10,11,11,11,11,11,11,11,11,11,10])
    start_time = time.time()
    NN = STAMP(data,512,None)
    print('TIME1: ',time.time()-start_time)
    np.save("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/testDataStamp.npy",NN)