from hmmlearn import hmm
import numpy as np
def get_hmms_ready(traindata,testData):
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
    results =[]
    for test in range(0,len(testhmm_matrix)):
        tempres =[]
        for train in range(0,len(trainhmm_matrix)):
            train1 =trainData[train].reshape(-1,1)
            test1 = testData[test].reshape(-1,1)
            a = 1/2*(trainhmm_matrix[train].score(test1)-trainhmm_matrix[train].score(train1) +
            testhmm_matrix[test].score(train1)-testhmm_matrix[test].score(test1))
            tempres.append(a)
        results.append(tempres)
    return results
def compute_hmm_values(trainData,testData,n_values_to_sum=5):
    trainhmm_matrix, testhmm_matrix = get_hmms_ready(trainData,testData)
    result =compare_hmms(trainhmm_matrix,testhmm_matrix,trainData,testData)
    result = np.array(result)
    result2 = []
    for x in result:
        ind = np.argpartition(x, n_values_to_sum)[:n_values_to_sum]
        temp = np.sum(x[ind])
        result2.append([temp])
    return np.array(result2)
def predict(output):
    i = np.mean(output)
    pred =[]
    for x in output.flat:
        if x<i:
            pred+=[0]
        else:
            pred+=[1]
    return pred
if __name__ =="__main__":
    #testing
    from gmuwork.shortcuts import quick_pfp1_file_reader
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:10]
    v01 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter01")[0:10]
    v10 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:10]
    v11 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter01")[0:10]
    v20 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:10]
    v21 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter01")[0:10]
    v30 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:10]
    v31 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter01")[0:10]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:10]
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:10]
    trainData = np.concatenate((v00,v10,v20,v30))
    testData = np.concatenate((v01,v11,v21,v31,v40,v41))
    l = [0]*40+[1]*20
    data_train, data_test, labels_train, labels_test=train_test_split(testData,l,train_size =0,random_state=4)
    result2 = compute_hmm_values(trainData,data_test)
    print(result2)
    print(np.mean(result2))
    print(len(result2))
    print(labels_test)
