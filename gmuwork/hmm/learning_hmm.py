from hmmlearn import hmm
import numpy as np 
from gmuwork.shortcuts import quick_pfp2_file_reader
if __name__ == "__main__":
    state1 = quick_pfp2_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeA")
    first = state1[0]
    model = hmm.GaussianHMM(n_components=1)
    model.fit(first.reshape(-1,1))
    print(model.score(state1[1].reshape(-1,1)))
