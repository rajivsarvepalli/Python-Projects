from compute_hmm_values import compute_hmm_values
import numpy as np
import warnings
warnings.filterwarnings("ignore") #for now beacuse depreacted warning machine
def getVectorData(percent_of_norm_files_in_train,n):
    from gmuwork.shortcuts import quick_pfp1_file_reader, cluster_based_over_under_sampling
    from sklearn.utils import shuffle
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:n]
    v01 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter01")[0:n]
    v10 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:n]
    v11 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter01")[0:n]
    v20 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:n]
    v21 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter01")[0:n]
    v30 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:n]
    v31 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter01")[0:n]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:n]
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:n]
    all_norm_data = np.concatenate((v00,v10,v20,v30,v01,v11,v21,v31))
    np.random.shuffle(all_norm_data)
    num_of_train_files = (int)(len(all_norm_data) * percent_of_norm_files_in_train)
    norm_data = all_norm_data[0:num_of_train_files]
    testData = np.concatenate((all_norm_data[num_of_train_files:len(all_norm_data)], v40, v41))
    l = [0]*(len(testData)-40)+[1]*40
    testData, l = cluster_based_over_under_sampling(testData,l,ratio=0.8)
    testData, l = shuffle(testData,l)
    return norm_data, testData, l
def save_hmm_values_to_file(file_name1, file_name2,n):
    norm_data, testData, l = getVectorData(.2,n)
    hmm_disatnces_aka_forest_data = compute_hmm_values(norm_data,testData)
    np.save(file_name1, hmm_disatnces_aka_forest_data)
    np.save(file_name2, l)
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    # data_train, data_test, labels_train, labels_test = train_test_split(hmm_disatnces_aka_forest_data,l,train_size =.5,random_state=4)
    # nb = GaussianNB()
    # nb.fit(data_train,labels_train)
    # pred = nb.predict(data_test)
    # print(accuracy_score(labels_test,pred))
    save_hmm_values_to_file('C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/hmm_data/hmm_values_of_40_per_file','C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/hmm_data/hmm_labels_of_40_per_file',40)
    