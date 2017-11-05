import numpy as np
import os
def load_then_to_txt(loc_of_np,loc_to_write):
    X = np.load(loc_of_np)
    X = np.real(X)
    print(X[0])
    f = open(loc_to_write,"w")
    f.write('''@RELATION stats\n\n@ATTRIBUTE mean  NUMERIC\n@ATTRIBUTE median  NUMERIC\n@ATTRIBUTE skewness  NUMERIC\n@ATTRIBUTE kurtosis  NUMERIC\n@ATTRIBUTE standard_dev  NUMERIC\n@ATTRIBUTE sumofmax  NUMERIC\n@ATTRIBUTE sumofmin  NUMERIC\n@ATTRIBUTE RMS  NUMERIC\n@ATTRIBUTE interquartile_range  NUMERIC\n@ATTRIBUTE l_scale  NUMERIC\n@ATTRIBUTE l_skewness  NUMERIC\n@ATTRIBUTE l_kurtosis  NUMERIC\n@ATTRIBUTE L4  NUMERIC\n@ATTRIBUTE L5  NUMERIC\n@ATTRIBUTE class        {normal,malware-infected}\n\n@DATA\n''')
    l = ['normal']*100+['malware-infected']*100
    l = np.vstack(l)
    X = np.hstack((X,l))
    X = np.array(X,dtype=object)
    s = ""
    for x in X:
        for y in x:
            s += str(y) + ","
        s = s[0:len(s)-1]
        s+="\n"
    f.write(s)
    f.close()
if __name__=="__main__":
    # load_then_to_txt("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/matrix_stats_with_fft.npy","C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/high_order_stats_txt_with_fft.arff")
    from gmuwork.shortcuts import numpyarr_to_arff_format_in_string
    X = np.load("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/matrix_stats_vector_dataset.npy")
    l = [0]*100+[1]*100
    l = np.vstack(l)
    X = np.hstack((X,l))
    s = numpyarr_to_arff_format_in_string(X,"stats",['mean  NUMERIC', 'median  NUMERIC', 'skewness  NUMERIC', 'kurtosis  NUMERIC', 
    'standard_dev  NUMERIC','sumofmax  NUMERIC','sumofmin  NUMERIC','RMS  NUMERIC','interquartile_range  NUMERIC', 
    'l-scale  NUMERIC', 'l-skewness  NUMERIC','lkurtosis  NUMERIC','L4  NUMERIC','L5  NUMERIC',['normal','malware-infected']])
    f = open("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/matrix_stats_vector_dataset.arff","w")
    f.write(s)
    f.close()
    print("Done")

