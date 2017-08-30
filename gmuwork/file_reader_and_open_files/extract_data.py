import os
import numpy as np
import json
import warnings
import time
class data_and_metadata:
    def __init__(self,glob,capture,annotations,source,data):
        self.core_global = glob
        self.core_capture =capture
        self.core_annotations = annotations
        self.PFP_source = source
        # for key, value in dictionary.items():  #automtic creation of every instance
        #     setattr(self, key, value)
        #     if isinstance(value,dict):
        #         for k,v in value.items():
        #             setattr(self, k, v)
        #     elif isinstance(value,list):
        #         for x in value:
        #             for k,v in x.items():
        #                 setattr(self, k, v)
        self.data = data
def extract_data(file_name,dt):
    data = np.fromfile(file_name,dtype=dt)
    return data
def extract_metadata(file_name):
    f = open(file_name,"r")
    return json.load(f)
def allData(directory_name):
    '''Takes a folder returns all the contents of the meta and data files as list of data_metadata objects;
    these objects contain 4 variables for each of the main namespaces, and each of these variables may 
    conatin dictionaries, or sometimes lists of dictionaries
    '''
    directory = os.fsencode(directory_name)
    list_of_obj =[]
    meta_files= []
    data_files =[]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".meta"): 
            meta_files+=[os.path.join(directory_name, filename)]
        elif filename.endswith(".data"):
            data_files+=[os.path.join(directory_name, filename)]
    
    for x in meta_files:
        s = str(os.path.splitext(x)[0])
        st = s+".data"
        try:
            i = data_files.index(st)
            dicti = extract_metadata(x)
            glob = dicti['core:global']
            capture = dicti['core:capture']
            annotations = dicti['core:annotations']
            PFP_source = dicti['PFP:source']
            dt = datatype(glob['core:datatype'])#data type processing
            list_of_obj+=[data_and_metadata(glob,capture,annotations,PFP_source,extract_data(data_files[i],dt))]
        except ValueError:
            raise ValueError("There is no matching datafile for the metadata file: " + s + ".meta")
        del data_files[i]
    if len(data_files) !=0:
        warnings.warn("There were " + str(len(data_files)) + " unused data files due to not having a matching metadata file")
    return list_of_obj
def datatype(st):
    strin =""
    rc = ""
    if st.endswith("_le"):
        strin+="little"
    else:
        strin+="big"
    i = st.index("_")
    fiu = None
    size = int(st[2:i])
    if st[0]=='r':
        if st[1] == 'f':
            if size == 16:
                fiu = np.float16
            elif size ==32:
                fiu = np.float32
            else:
                fiu = np.float_
        elif st[1] == 'i':
            if size == 8:
                fiu = np.int8
            elif size == 16:
                fiu = np.int16
            elif size ==32:
                fiu = np.int32
            else:
                fiu = np.int64
        else:
            if size == 8:
                fiu = np.uint8
            elif size == 16:
                fiu = np.uint16
            elif size ==32:
                fiu = np.uint32
            else:
                fiu = np.uint64
    else:
        fiu = np.complex
    return np.dtype([(strin,fiu)])
        
if __name__ == "__main__":
    st =time.time()
    x =allData(r"C:\Users\Rajiv Sarvepalli\Projects\Python-Projects\GMU Work\AllData\SIGMF_format_data\Picos3406D")
    print(time.time()-st)
    # for i in range(0,len(x)):
    #     for key in vars(x[i]):
    #         s= vars(x[i])[key]
    #         print(str(key) + ": "+ str(s))
    #         print("\n")
    print("Done")