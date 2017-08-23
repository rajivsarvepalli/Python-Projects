import numpy as np
import os
from pfp2_file_reader import getData,pfp2_files_data
def extract(file_name):
    file_object = open(file_name,"r")
    lin =file_object.read()
    line = lin.replace("\n","\t")#add
    data = np.array(line.split("\t"))
    data = np.delete(data, data.size-1)
    file_object.close()
    return data
def allData(directory):
    txtfiles=[]
    allfiles = []
    numOfUnusedFiles =0 #debugging variable
    directories = os.listdir(directory)
    for infile in directories:
        temp = infile.find(".")
        if temp !=-1:
            index = len(infile)-temp
            file_extension = infile[-index:]
            if file_extension==".pfp2":
                z=getData(directory +"/" + infile)
                allfiles.append(z)
            elif file_extension==".txt":
                z = extract(directory +"/" + infile)
                txtfiles.append(z)
            else:
                numOfUnusedFiles+=1
    print("numOfUnusedFiles: " + str(numOfUnusedFiles))
    matrix = np.array(txtfiles)
    allfiles.append(matrix)
    return allfiles
def writeAllData(directory,file_to_write):
    complete = allData(directory)
    f = open(file_to_write,"w")
    numofPfpFiles = 0
    for x in complete:
        if isinstance(x,pfp2_files_data):
            numofPfpFiles+=1
            for a in x.data:
                f.write(str(a)+"\t")
        else:
            break
    f.write('\n'.join('\t'.join(x for x in y) for y in complete[numofPfpFiles]))
    f.close()
#end
#testing
#x=allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/fakedata/Combination_of_pfp2_and_txt")#opeartion is about 1/10 second
#writeAllData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/fakedata/Combination_of_pfp2_and_txt",
#"C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/writehere.txt")#between 8 and 9 seconds
