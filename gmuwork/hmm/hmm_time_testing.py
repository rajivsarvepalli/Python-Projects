from gmuwork.shortcuts import time_function
import warnings
warnings.filterwarnings("ignore") #for now beacuse depreacted warning machine
import numpy as np
from compute_hmm_values import compute_hmm_values
from gmuwork.shortcuts import quick_pfp1_file_reader
from gmuwork.shortcuts import simple_line_graph_with_points
def profile():
    import scipy.io as sio
    from gmuwork.shortcuts import quick_pfp2_file_reader
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:40]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:40]
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:40]
    graphviz = GraphvizOutput(output_file='C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/hmm_data/profile_compute_hmm_values_function.png')
    with PyCallGraph(output=graphviz):
        compute_hmm_values(v00,np.concatenate((v40,v41)))
def time_compute_hmm_for_varying_ns():
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")
    times = []
    for i in range(15,170,10):
        times += [time_function(compute_hmm_values, v00[0:i], np.concatenate((v40[0:i],v41[0:i])))]
    simple_line_graph_with_points([15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165],times,annotated_values=(True,2))
if __name__ == "__main__":
    time_compute_hmm_for_varying_ns()
    
    
