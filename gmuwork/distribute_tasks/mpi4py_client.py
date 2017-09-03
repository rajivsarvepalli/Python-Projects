from gmuwork.shortcuts import STAMP
import mpi4py
def mpi4py_helper():
    comm = mpi4py.MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()
def with_mpi4py():
    print("hi")
if __name__ =="__main__":
    mpi4py_helper()
    print("Done")