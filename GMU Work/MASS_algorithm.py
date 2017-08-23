import numpy as np
# def movstd2(x,m):
#     x = x.astype("float")
#     s = np.insert(np.cumsum(x), 0, 0)
#     sSq = np.insert(np.cumsum(x ** 2), 0, 0)
#     segSum = s[m:] - s[:-m]
#     segSumSq = sSq[m:] - sSq[:-m]
#     return np.sqrt(segSumSq / m - (segSum / m) ** 2)
# def findNN2(x,y):
#     x = np.array(x)
#     y = (y-np.mean(y))/np.std(y)
#     m = len(y)
#     n = len(x)
#     stdv = movstd2(x, m)
#     y = y[::-1]
#     y = np.pad(y, (0, n - m), 'constant')
#     dots = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y))
#     return np.sqrt(2 * (m - (dots[m - 1 :] / stdv)))
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
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    Z = X*Y
    z = np.fft.irfft(Z)
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
def STAMP(time_seriesA,m,time_seriesB=None):
    if time_seriesB == None:
        time_seriesB = time_seriesA
    nB = len(time_seriesB)
    P = np.array((nB-m)*[np.inf])
    I = np.array((nB-m)*[0])
    for idx in range(0,(nB-m)):
        D = findNN(time_seriesA,time_seriesB[idx:idx+m])
        # for i in range(0,nB-m):
        #     if D[i]<=P[i]:
        #         P[i] = D[i]
        #         I[i] = idx
    return P,I
if __name__ =="__main__":
    #testing
    print(z)
    y = STAMP([1,2,3,4,7,8,9,9,9,9,9,23,343,222,11,333,5553,333,3222,32233],3)
    print(y)