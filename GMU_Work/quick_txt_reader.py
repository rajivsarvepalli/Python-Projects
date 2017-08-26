import os
import numpy as np
def extract(file_name):
    file_object = open(file_name,"r")
    lin =file_object.read()
    line = lin.replace("\n","\t")#add
    line = line.replace(" ", "")
    z = [x for x in line.split("\t") if x]

    data = np.array(z,dtype=float)
    file_object.close()
    return data
def allData(directory):
    txtfiles=[]
    directories = os.listdir(directory)
    for infile in directories:
        z = extract(directory +"/" + infile)
        txtfiles.append(z)  
    return np.array(txtfiles)