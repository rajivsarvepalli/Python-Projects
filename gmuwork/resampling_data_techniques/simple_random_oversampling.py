from collections import Counter
import numpy as np
from random import randint
import random
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
def random_oversampling(X,y,ratio=.5):
    '''
    Returns new dataset and labels with random data vectors of the minority class added
    Parameters
    ----------
    X : arraylike dataset
    y : arraylike labels for dataset
    ratio : number of minoirty class to number of majority class
    Returns
    ----------
    X : new dataset with random portions of minority data vectors added to old dataset
    y : new labels for new dataset 
    '''
    minority_class = find_minority_class(y)
    majority_class = find_majority_class(y)
    #(num_of_minority_repeats+x)/num_of_majority_repeats = ratio
    num_of_syntheic_samples = int((ratio*majority_class[1])-minority_class[1])
    syntheic_samples = []
    for i in range(len(y)):
        if y[i] == minority_class[0]:
            syntheic_samples.append(X[i])
    np.random.shuffle(syntheic_samples)
    if len(syntheic_samples)>=num_of_syntheic_samples:
        syntheic_samples = syntheic_samples[0:num_of_syntheic_samples]
    else:
        while len(syntheic_samples)<num_of_syntheic_samples:
            syntheic_samples.append(syntheic_samples[randint(0,len(syntheic_samples)-1)])
    syntheic_samples = np.array(syntheic_samples)
    return np.concatenate((syntheic_samples,X)), np.insert(y,0,[minority_class[0]]*num_of_syntheic_samples)
def random_undersampling(X,y,ratio=.5):
    '''
    Returns new dataset and labels with random data vectors of the majority class removed
    Parameters
    ----------
    X : arraylike dataset
    y : arraylike labels for dataset
    ratio : number of minority class to number of majority class
    Returns
    ----------
    X : new dataset with random portions of majority data vectors
        removed from old dataset
    y : new labels for new dataset 
    '''
    minority_class = find_minority_class(y)
    majority_class = find_majority_class(y)
    #(num_of_minority_repeats)/(num_of_majority_repeats-x) = new_ratio
    #-(num_of_minority_repeats)/new_ratio + num_of_majority_repeats = x
    num_of_samples_to_remove = (int)(-(minority_class[1])/ratio + majority_class[1])
    index_to_remove = []
    random_indexes = random.sample(range(len(X)), len(X))
    i = 0
    while num_of_samples_to_remove > 0:
        if y[random_indexes[i]] == majority_class[0]:
            num_of_samples_to_remove = num_of_samples_to_remove-1
            index_to_remove +=[random_indexes[i]]
        i = i+1
            
    X = np.delete(X,index_to_remove,axis=0)
    y = np.delete(y,index_to_remove)
    return X,y
if __name__ == "__main__":
    data = []
    l= []
    for i in range(0,100):
        data.append([70+i,70+i])
        l+=[1]
        if i%10 == 0:
            data.append([(2+(i/1000)),(2+(i/1000))])
            l+=[0]
    
    l = np.array(l)
    data = np.array(data)
    #data = np.reshape(data,(-1,1))
    print(data)
    x = data
    y = l
    for i in range(0,10000):
        x,y = random_undersampling(x,y)

