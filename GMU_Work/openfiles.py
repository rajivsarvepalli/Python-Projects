import numpy as np
import os
import time
from findFile import find
from pfp2_file_reader import getData
#extracts the file's data as a 1d array
def extract1d(file_name):
    file_object = open(file_name,"r")
    lin =file_object.read()
    line = lin.replace("\n","\t")
    data = np.array(line.split("\t"))
    data = np.delete(data, data.size-1)
    file_object.close()
    return data
#extracts the file's data as a 2d array
def extract2d(file_name):
    file_object = open(file_name,"r")
    line =file_object.read()
    data = np.array(line.split("\t"))
    file_object.close()
    return data
#collects all the data from inside folder and uses extract1d; returns array of all the data
def allData():
    mylist=[]
    for infile in os.listdir("directory of folder"):
        z = extract1d("directory of folder" + infile)
        mylist.append(z)
    matrix = np.array(mylist)
    return matrix
#collects all the data from inside folder and uses extract2d; returns list of arrays containing data
#also prints times of loadinf files into memory
def specificData(location,file_extension):
    mylist=[]
    times=[]
    for infile in os.listdir(location):
        if file_extension==".txt":
            start_time =time.time()
            z = extract2d(location + infile)
            mylist.append(z)
            elapsed_time = time.time()-start_time
            times.append(elapsed_time)
            #print or save times to a file to view
        if file_extension == ".pfp2":
            start_time =time.time()
            z = getData(location + infile)
            mylist.append(z)
            elapsed_time = time.time()-start_time
            times.append(elapsed_time)
    print(times)
    return mylist

#testing
# complete=specificData("C:",".pfp2")
# for z in complete:
#     st = ('\t'.join(x for x in z))
#     print(st+"\n")
# filec = open(find("b = '\n'.join('\t'.join(x for x in y) for y in complete)writehere.txt","C:/USers/RAjiv SArvepalli"),"w") 

# filec.write(b)
# filec.close()
