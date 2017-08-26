import numpy as np
def MASS(time_series,query):
    '''
    input: a time_series and query\n
    compares the time series to query \n
    and returns A distance profile of the query
    '''
    #ca
    QT = SlidingDotProduct(time_series,query)#computes sliding dot product of the time series and query
    m = len(query)
     #μQ is the mean of Q,
     # MT[i] is the mean of Ti,m, 
     # σQ is the standard deviation of Q
     #ΣT[i] is the standard deviation of Ti,m.
     #compute those values
     #then use function CalculateDistanceProfile(Q, T, QT, μQ, σQ, ΜT, ΣT) and return that 
def STAMP():
    '''

    '''
def SlidingDotProduct(time_series,query):
    '''
    compares the time series to uery\n
    returns dot product in q and all subsequences in time_series
    '''
    n =len(time_series)
    m=len(query)
    query = query[::-1] #reverse query
    time_series = np.append(time_series,[0]*n) #append time series with n zeroes
    query = np.append(query,[0]*(2*n-m)) #append reversed query with 2n-m zeroes
    qfft = np.fft.fft(query)  #fft of query
    tfft = np.fft.fft(time_series)  #fft of time series
    return np.fft.ifft(np.multiply(qfft,tfft))  #inverse fft of element-wsie multiplication of qfft and tfft

if __name__ == "__main__":
    print((movstd2(np.array([1,2,3,4]), 3)))
    