from hmmlearn import hmm
import numpy as np
def prepare_hmms(traindata,testData):
    '''
    make hidden markov models trained for each vector of the dataset
    returning two arrays one array of the hmms for the trainData, 
    and the other hmms for the testData
    '''
    testhmm_matrix =[]
    trainhmm_matrix = []
    print(len(traindata))
    for i in range(0,len(traindata)):
        model = hmm.GaussianHMM(n_components=1)
        model.fit(traindata[i].reshape(-1,1))
        trainhmm_matrix.append(model)
    for i in range(0,len(testData)):
        model = hmm.GaussianHMM(n_components=1)
        model.fit(testData[i].reshape(-1,1))
        testhmm_matrix.append(model)
    return np.array(trainhmm_matrix),np.array(testhmm_matrix)
def compare_hmms(trainhmm_matrix,testhmm_matrix,trainData,testData):
    '''
    For each test hidden markov model compare to each train markov model using formula:\n
        Distance = 1/2 * TRHM.sc(TEV) - TRHM.sc(TRV) + TEHM.sc(TRV) - TEHM.sc(TRV)
    Note: \n
    TRHM = trainhmm\n
    TEHM = testhmm\n
    .sc() = score()\n
    TRV = train vector\n
    TEV = test vector
    '''
    results =[]
    for test in range(0,len(testhmm_matrix)):
        temp_result =[]
        for train in range(0,len(trainhmm_matrix)):
            train_vector =trainData[train].reshape(-1,1)
            test_vector = testData[test].reshape(-1,1)
            a = 1/2*(trainhmm_matrix[train].score(test_vector)-trainhmm_matrix[train].score(train_vector) +
            testhmm_matrix[test].score(train_vector)-testhmm_matrix[test].score(test_vector))
            temp_result.append(a)
        results.append(temp_result)
    return results
def compute_hmm_values(trainData,testData,n_values_to_sum=5):
    '''
    input: trainData, testData and the number of values to sum (n-nearest neighbors)
    output: the n nearest output values summed of each 
    data vector in the dataset(dataset is both trainData and testData)
    '''
    trainhmm_matrix, testhmm_matrix = prepare_hmms(trainData,testData)
    result =compare_hmms(trainhmm_matrix,testhmm_matrix,trainData,testData)
    result = np.array(result)
    result2 = []
    for x in result:
        ind = np.argpartition(x, n_values_to_sum)[:n_values_to_sum]
        temp = np.sum(x[ind])
        result2.append([temp])
    return np.array(result2)
if __name__ =="__main__":
    #testing
    from gmuwork.shortcuts import quick_pfp1_file_reader
    from sklearn.model_selection import train_test_split
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:20]
    v01 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter01")[0:20]
    v10 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:20]
    v11 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter01")[0:20]
    v20 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:20]
    v21 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter01")[0:20]
    v30 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:20]
    v31 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter01")[0:20]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:20]
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:20]
    trainData = np.concatenate((v00,v10,v20,v30))
    testData = np.concatenate((v01,v11,v21,v31,v40,v41))
    l = [0]*80+[1]*40
    data_train, data_test, labels_train, labels_test = train_test_split(testData,l,train_size =0,random_state=4)
    result2 = compute_hmm_values(trainData,data_test)
    print(result2)
    print(np.mean(result2))
    print(len(result2))
    print(labels_test)
