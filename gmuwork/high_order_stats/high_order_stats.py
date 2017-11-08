import lmoments3 as lm
import numpy as np
from scipy import stats
from sklearn import metrics
def onerowstats(x,n):
    return np.array([np.mean(x),np.median(x),stats.skew(x),stats.kurtosis(x),np.std(x),sumofmaxs_or_mins(x,10),sumofmaxs_or_mins(x,10,max_or_min="min"),RMS(x),interquartile_range(x),*lstats(x,n)])
def matrix_stats(X,n=6):
    '''
    Order of stats: mean, median, skewness, kurtosis, standard dev,sumofmax,sumofmin,RMS,interquartile_range, l-scale, l-skewness,lkurtosis
    ^that is if n=4, if not 4 the order will be the same but additonal L-moments will added to the end, 1 per each value of n that increases
    Input: Matrix of rows of data 
    Output: 'summary' of each in row vectors joined into a matrix,
    basically a matrix of stats summarizing the data matrix
    '''
    X = np.array(X)
    stats = []
    try:
        for x in X:
            stats.append(onerowstats(x,n))
    except TypeError:
        stats.append(onerowstats(X,n))
    return np.array(stats)
def lkurtosis(x):
    return lm.lmom_ratios(x)[3]
def lskewness(x):
    return lm.lmom_ratios(x)[2]
def lscale(x):
    return lm.lmom_ratios(x)[1]
def mean(x):
    return lm.lmom_ratios(x)[0]
def sumofmaxs_or_mins(x,n,max_or_min="max"):
    if max_or_min == "min":
        index = np.argpartition(x,-n)[:-n]
        return np.sum(x[index])
    elif max_or_min == "max":
        index = np.argpartition(x,n)[n:]
        return np.sum(x[index])
    else:
        raise ValueError("max_or_min must equal exactly either 'max' or 'min'")
def interquartile_range(x):
    '''
    Input: arraylike x

    Output: the interquartile range of x


    '''
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr
def RMS(x):
    '''
    Input: arraylike x

    Output: the root mean square of x


    '''
    return np.sqrt(mean(x**2))
def lstats(x,n):
    '''
    n is the number of  the lratios you wish to include, starting at L2 to exclude mean
    the function return L2,L3,L4 if n=4
    '''
    lratios = lm.lmom_ratios(x,nmom=n)
    return lratios[1:n]
def profile(data):
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    data = data
    graphviz = GraphvizOutput(output_file='C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/profile_STATS.png')
    with PyCallGraph(output=graphviz):
        matrix_stats(data)
if __name__ =="__main__":
    #testing 
    import time
    from numpy.fft import fft
    from gmuwork.shortcuts import quick_pfp2_file_reader
    s1 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:25]
    s2 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:25]
    s3 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:25]
    s4 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:25]
    sT = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:100]
    d = np.concatenate((s1,s2,s3,s4,sT),axis=0)
    d = d.astype(np.float64)
    d = np.fft.rfft(d)
    start_time = time.time()
    y = matrix_stats(d)
    print(time.time()-start_time)
    np.save("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/matrix_stats_vector_dataset_With_fft.npy",y)