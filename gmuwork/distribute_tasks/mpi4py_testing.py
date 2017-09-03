from gmuwork.shortcuts import STAMP#stamp import here
import numpy as np
from gmuwork.shortcuts import quick_txt_reader
import time
def scatter(func,objects):
    from mpi4py import MPI
    '''
    set up right now for my STAMP function
    objects is tuple of arguments
    len(objects[0])/number of processes must be >= 2
    comm is an mpi comm
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank ==0:
        elments_pre_proc = len(objects[0])//size
        data = np.array(objects[0])
        split_data = []
        for x in range(size-1):
            split_data.append(data[x*elments_pre_proc:(x+1)*elments_pre_proc])
        split_data.append(data[(size-1)*elments_pre_proc:len(data)])
        data = split_data
    else:
        data =None
    data = comm.scatter(data,root=0)
    comm.bcast(func,root=0)
    m = objects[1]
    tB = objects[2]
    result = []
    for i in range(len(data)):
        result.append(func(data[i],m,tB))
    res = comm.gather(result,root=0)
    return res
a = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeA")
b = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeB")
c = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeC")
d = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeD")
m = quick_txt_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/KMeans_training/ModeM")
d = np.concatenate((a,b,c,d,m))
print("starting...stamp")
start_time = time.time()
r = scatter(STAMP,[d[0:8],256,None])
print(time.time()-start_time)
print(r)
print(len(r))
print(len(r[0]))
