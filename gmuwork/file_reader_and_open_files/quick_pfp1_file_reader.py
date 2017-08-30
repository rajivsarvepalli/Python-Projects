import os
import numpy as np

def extract(file_name):
    f= open(file_name,"rb")
    f.seek(50)
    a = np.fromfile(f,dtype=np.single)
    return a
def alldata(directory):
    files=[]
    for infile in os.listdir(directory):
        files.append(extract(directory + "/" + infile))
    return np.array(files)
if __name__ == "__main__":
    print(alldata(r"C:\Users\Rajiv Sarvepalli\Projects\Python-Projects\GMU Work\AllData\dataSet3\Vector0001Path0001Iter01"))