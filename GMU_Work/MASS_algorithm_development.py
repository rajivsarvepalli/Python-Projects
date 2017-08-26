import numpy as np
from scipy.fftpack import fft,ifft
import pyfftw
def MASSPRE(x,m):
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
    Implementation of the MASS algorithm
    input: a time_series x, and query y
    output: 
    '''
    #normlaize line below
    y = (y-np.mean(y))/np.std(y)

    m = len(y)
    y = y[::-1]
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y,2))
    y = np.append(y,((n)-(m))*[0])
    Y = fft(y)
    Z = X*Y
    Z = pyfftw.n_byte_align(Z,None)
    z = pyfftw.interfaces.scipy_fftpack.ifft(Z)
    dist = value - 2*(z[m:n] - sumy*meanx)/sigmax + sumy2
    return np.abs(np.lib.scimath.sqrt(dist))
def findNN(x,y):
    '''
    Implementation of the MASS algorithm
    input: a time_series x, and query y
    output: 
    '''
    n = len(x)
    #normlaize line below
    y = (y-np.mean(y))/np.std(y)

    m = len(y)
    x = np.append(x,(n)*[0])
    y = y[::-1]
    y = np.append(y,((2*n)-(m))*[0])
    X = fft(x)
    Y = fft(y)
    Z = X*Y
    z = ifft(Z)
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y,2))
    #movstd start
    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(np.power(x,2))
    sumx2 = cum_sumx2[m:n] -cum_sumx2[0:n-m]
    sumx = cum_sumx[m:n]-cum_sumx[0:n-m]
    meanx = np.divide(sumx,m)
    sigmax2 = (sumx2/m)- np.power(meanx,2)
    sigmax = np.sqrt(sigmax2)
    #movstd end
    dist = (sumx2 - 2*sumx*meanx + m*(np.power(meanx,2)))/sigmax2 - 2*(z[m:n] - sumy*meanx)/sigmax + sumy2
    return np.abs(np.lib.scimath.sqrt(dist))
def STAMP(time_seriesA,m,time_seriesB):
    '''
    set time_SeriesB to none if compare A to itself
    '''
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
def profile():
    import scipy.io as sio
    from STAMP import STAMP as stmp
    from shortcuts import quick_pfp2_file_reader
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    from daniels_algorithm import test2
    matfile = sio.loadmat("C:/Users/Rajiv Sarvepalli/Downloads/testData.mat")
    data = matfile['data'][0]
    graphviz = GraphvizOutput(output_file='C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/profile_STAMP.png')
    with PyCallGraph(output=graphviz):
        stmp(data,2048,None)
if __name__ =="__main__":
    #testing
    import time
    import scipy.io as sio
    from shortcuts import quick_txt_reader
    #profile()
    matfile = sio.loadmat("C:/Users/Rajiv Sarvepalli/Downloads/testData.mat")
    data = matfile['data'][0]
    start_time =time.time()
    findNN(data,data[0:200])
    print('TIME1: ',time.time()-start_time)
    # for x in range(0,142):
    #     data+=[2,2,2,2,2,5,5,5,5,5,6,6,6,6,6,6,8,8,8,8,8,9,9,9,9,9,8,8,8,8,8,6,6,6,6,6,5,5,5,5,5,2,2,2,2,2]
    # data = np.insert(data,2500,[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10])
    # data = np.insert(data,1000,[10,10,10,10,10])
    # data = np.insert(data,4500,[10,11,11,11,11,11,11,11,11,11,10])
    start_time = time.time()
    NN = STAMP(data,200,None)
    print('TIME1: ',time.time()-start_time)
    np.save("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/testDataStamp.npy",NN)
    




