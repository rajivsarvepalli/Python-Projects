import numpy as np
from sklearn.metrics import accuracy_score
import os
import numpy as np
def extract(file_name):
    f= open(file_name,"r")
    a = np.fromfile(f,dtype=np.single)
    data =a[14:len(a)]
    return data
def alldata(directory):
    files=[]
    for infile in os.listdir(directory):
        files.append(extract(directory + "/" + infile))
    return np.array(files)
if __name__ == "__main__":
    print(alldata(r"C:\Users\Rajiv Sarvepalli\Projects\Python-Projects\GMU Work\AllData\dataSet2\State2"))


