import numpy as np
import matplotlib.pyplot as plt 
from gmuwork.shortcuts import STAMP, moving_mean_smoothing, quick_pfp1_file_reader
if __name__ == "__main__":
    s1 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[5]
    sT = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0]
    s1 = moving_mean_smoothing(s1,500)
    sT = moving_mean_smoothing(sT,500)
    times = [x for x in range(0,len(s1))]
    stamp1 = STAMP(s1,256,None)
    stamp2 = STAMP(sT,256,None)
    print('State1 Stamp Mean',np.mean(stamp1[0]))
    print('StateTamper Stamp Mean',np.mean(stamp2[0]))


