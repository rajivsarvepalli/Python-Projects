from gmuwork.shortcuts import using_hmms_to_compute_summed_distances
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
import logging
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
def getVectorData(percent_of_norm_files_in_train):
    from gmuwork.shortcuts import quick_pfp1_file_reader
    from sklearn.utils import shuffle
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
    all_norm_data = np.concatenate((v00,v10,v20,v30,v01,v11,v21,v31))
    np.random.shuffle(all_norm_data)
    num_of_train_files = (int)(len(all_norm_data) * percent_of_norm_files_in_train)
    norm_data = all_norm_data[0:num_of_train_files]
    testData = np.concatenate((all_norm_data[num_of_train_files:len(all_norm_data)],v40, v41))
    l = [0]*(len(testData)-40)+[1]*40
    testData, l = shuffle(testData,l)
    return norm_data, testData, l
def roc_curve(distances,mean,labels_test):
    fpr =[]
    tpr =[]
    for i in range(0,400):
        x = compare_to_mean(distances,mean,i/100)
        cm = confusion_matrix(labels_test,x)
        print(i,accuracy_score(labels_test,x))
        fpr.append(1-(cm[0][0]/(cm[0][0]+cm[0][1])))
        tpr.append(cm[1][1]/(cm[1][0]+cm[1][1]))
    tpr = np.nan_to_num(tpr)
    fpr = np.nan_to_num(fpr)
    # print(fpr)
    # print("\n\n\n\n\n")
    # print(tpr)
    roc_auc = auc(fpr,tpr)
    print("Auc: "+ str(auc(fpr,tpr)))
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    for i in range(0,400,5):
         plt.annotate(str(i),(fpr[i],tpr[i]))
    # for i in range(97,100,1):
    #      plt.annotate(str(i),(fpr[i],tpr[i]))
         #print(str(i)+ " : " + str(fpr[i])+ ","+str(tpr[i]))
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-.02,1.02])
    plt.ylim([0,1.4])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver operating characteristic Dataset2') #name of current dataset that you are testing
def compare_to_mean(distnaces, mean, c):
    check_val = mean*c
    pred = []
    for x in distnaces:
        if x[0] > check_val:
            pred+=[0]
        else:
            pred+=[1]
    return pred
if __name__ == "__main__":
    from gmuwork.shortcuts import quick_pfp1_file_reader
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from gmuwork.shortcuts import confusion_matrix_plotter, memory_usage_psutil
    from gmuwork.shortcuts import ADASYN, borderlineSMOTE, simple_random_oversampling, simple_random_undersampling, cluster_based_over_under_sampling
    distances = np.load('C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/hmm_data/hmm_values_of_40_per_file.npy')
    l = np.load('C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/hmm_data/hmm_labels_of_40_per_file.npy')
    all_data,newlabels = distances,l
    # safe,syntheic,danger = borderlineSMOTE(distances,l,1,4000,4)
    # all_data = np.concatenate((distances,safe,syntheic,danger))
    #all_data,newlabels = ADASYN(distances,l,ratio=0.7,imb_threshold=.7,k=10)
    #all_data,newlabels = cluster_based_over_under_sampling(distances,l,n_majority=6,n_minority=2,ratio=.8)
    # for i in range(0,len(distances)):
    #     print(distances[i])
    #     print(l[i])
    data_train, data_test, labels_train, labels_test = train_test_split(all_data, newlabels, train_size = .8, random_state=4)
    svc = GaussianNB()
    svc.fit(data_train,labels_train)
    print(labels_train)
    pred = svc.predict(data_test)
    confusion_matrix_plotter(labels_test, pred,['N','M'])
    print(accuracy_score(labels_test,pred))
    print('recall score', recall_score(labels_test,pred))
    mean = (np.mean(all_data))
    pred = compare_to_mean(all_data,mean, 4.99)
    print(l)
    l= np.array(l)
    plt.figure()
    roc_curve(all_data,mean,newlabels)
    plt.show()