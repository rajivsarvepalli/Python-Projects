import numpy as np
from collections import Counter
from sklearn.cluster import k_means
def find_minority_class(y):
    '''
    input: arraylike y of labels
    output: minoirty class tuple of (label, number of times) 
    '''
    y = Counter(y)
    most_common = y.most_common()
    return most_common[len(most_common)-1]
def find_majority_class(y):
    '''
    input: arraylike y of labels
    output: majority class tuple of (label, number of times) 
    '''
    y = Counter(y)
    most_common = y.most_common()
    return most_common[0]
def getMinority_Majority_Data(X,y):
    idx_min = []
    idx_maj = []
    minority_class = find_minority_class(y)
    majority_class = find_majority_class(y)
    for i in range(len(X)):
        if y[i]==minority_class[0]:
            idx_min+=[i]
        elif y[i]==majority_class[0]:
            idx_maj+=[i]
    return X[idx_min], X[idx_maj]
def form_clusters(X,n):
    centroids,labels,interia = k_means(X,n)
    return labels
def cluster_based_oversampling(X,y,n_majority,n_minority,ratio=.5):
    '''
    Returns new dataset and labels with clustered data vectors of 
    the minority class added and clustered data vectors of 
    the majority class removed, creating a new dataset 
    with relatively same length as original
    Parameters
    ----------
    X : arraylike dataset
    y : arraylike labels for dataset
    n_majority : number of majority clusters
    n_minority : number of minority clusters
    ratio : number of minority class to number of majority class
    Returns
    ----------
    X : new dataset of nearly the same length as original X
    y : new labels for the new dataset\n
    Returns x and y in form of x, y
    
    '''
    minority_class = find_minority_class(y)
    majority_class = find_majority_class(y)
    num_of_syntheic_samples = int((minority_class[1]-ratio*majority_class[1])*(1/(-1-ratio)))
    if num_of_syntheic_samples<0:
        ValueError("the minority class already has higher than wanted ratio")
    num_of_synthetic_to_add_to_min_clusters = (int)((num_of_syntheic_samples/n_minority))
    num_of_synthetic_to_remove_from_maj_clusters = (int)((num_of_syntheic_samples/n_majority))
    cluster_tracker_maj = np.array([num_of_synthetic_to_remove_from_maj_clusters]*n_majority)
    cluster_tracker_min = np.array([num_of_synthetic_to_add_to_min_clusters]*n_minority)
    min_data, maj_data = getMinority_Majority_Data(X,y)
    labels_min = form_clusters(min_data,n_minority)
    labels_maj = form_clusters(maj_data,n_majority)
    #majority removal first
    index_to_remove_from_maj = []
    for n in range(n_majority):
        for i in range(len(labels_maj)):
            if labels_maj[i] == n:
                index_to_remove_from_maj+=[i]
                cluster_tracker_maj[n] = cluster_tracker_maj[n]-1
            if (cluster_tracker_maj[n]==0):
                break
    #minoiryt
    items_to_add_to_min = []
    for n in range(n_minority):
        i =0
        while i<len(labels_min):
            if labels_min[i] == n:
                items_to_add_to_min.append(min_data[i])
                cluster_tracker_min[n] = cluster_tracker_min[n]-1
            if cluster_tracker_min[n] ==0:
                break
            else:
                if i == len(labels_min)-1:
                    i=0
            i = i+1

    maj_data = np.delete(maj_data,index_to_remove_from_maj,axis=0)
    X = np.concatenate((min_data,items_to_add_to_min,maj_data))
    y = [minority_class[0]]*(len(min_data)+len(items_to_add_to_min))+[majority_class[0]]*len(maj_data)
    return X, y
if __name__ == "__main__":
    data =[]
    l= []
    for i in range(0,500):
        data.append([70+i,70+i])
        l+=[1]
        if i%10 == 0:
            data+=[[2+(i/1000),2+(i/1000)]]
            l+=[0]
    l = np.array(l)
    data = np.array(data)
    #data = np.reshape(data,(-1,1))
    print(data)
    x = data
    y = l
    x,y = cluster_based_oversampling(x,y,4,2,ratio=1)
    print("done")
    
            


