from hmmlearn import hmm
import numpy as np
def prepare_hmms(traindata,testData):
    '''
    Returns trainhmm_matrix, testhmm_matrix, matrixs of one hmm per data vector for each portion of the dataset (trainData, testData)
    Parameters
    ----------
    trainData : arraylike training dataset (normal data)
    testData : arraylike test dataset (all other data)
    Returns
    ----------
    trainhmm_matrix : arraylike matrix of one hmm per data vector of the training dataset (the normal data)
    testhmm_matrix : arraylike matrix of one hmm per data vector of the test dataset
    Note
    ---------- 
    '''
    testhmm_matrix =[]
    trainhmm_matrix = []
    for i in range(0,len(traindata)):
        model = hmm.GaussianHMM(n_components=1)
        model.fit(traindata[i].reshape(-1,1))
        trainhmm_matrix.append(model)
    for i in range(0,len(testData)):
        model = hmm.GaussianHMM(n_components=1)
        model.fit(testData[i].reshape(-1,1))
        testhmm_matrix.append(model)
    return np.array(trainhmm_matrix), np.array(testhmm_matrix)
def compare_hmms(trainhmm_matrix,testhmm_matrix,trainData,testData):
    '''
    Returns smoothed dataset of X by computing moving mean
    Parameters
    ----------
    trainhmm_matrix : arraylike matrix of one hmm per data vector of the training dataset (the normal data)
    testhmm_matrix : arraylike matrix of one hmm per data vector of the test dataset
    Returns
    ----------
    hmm_distances : computed distance values, Matrix of dimensions are: len(testData), len(trainData); (Row*Column)
    Note
    ---------- 
    Description : For each test hidden markov model compare to each train markov model using formula:\n
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
def compute_hmm_values(trainData,testData,n=5):
    '''
    Returns biggest n hmm distance values summed per data vector\n
    Therefore 1 value per each data vector
    Parameters
    ----------
    trainData : the training data (the normal data only non-tampered state)
    testData : the rest of the data
    Returns
    ----------
    hmm_summed_distances : dataset of the top n computed hmm distances summed, one value per data vector of original dataset
    Note
    ----------
    Description : given some normality, compute_hmm_values can compare that to testData and see its relative distance to the normality
    '''
    trainhmm_matrix, testhmm_matrix = prepare_hmms(trainData,testData)
    result =compare_hmms(trainhmm_matrix,testhmm_matrix,trainData,testData)
    result = np.array(result)
    result2 = []
    for x in result:
        ind = np.argpartition(x, n)[:n]
        temp = np.sum(x[ind])
        result2.append([temp])
    return np.array(result2)
if __name__ =="__main__":
    #testing
    from gmuwork.shortcuts import quick_pfp1_file_reader
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
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
    print(labels_test)
    naiveb_test_data = result2[len(result2)-20:len(result2)]
    naiveb_train_data = result2[0:len(result2)-20]
    naiveb_test_data_labels = labels_test[len(labels_test)-20:len(labels_test)]
    naiveb_train_data_labels = labels_test[0:len(labels_test)-20]
    nb = GaussianNB()
    nb.fit(naiveb_train_data,naiveb_train_data_labels)
    pred = nb.predict(naiveb_test_data)
    print(accuracy_score(pred,naiveb_test_data_labels))

