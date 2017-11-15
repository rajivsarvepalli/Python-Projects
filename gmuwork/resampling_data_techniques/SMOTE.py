import numpy as np
from sklearn.neighbors import NearestNeighbors 
from random import randrange, choice
def SMOTE(T, N, k, h = 1.0):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.
    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 
    Returns
    -------
    S : Synthetic samples. array, 
        shape = [(N/100) * n_minority_samples, n_features]. 
    """    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = N/100
    n_synthetic_samples = int(N * n_minority_samples)
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    N = int(N)
    #Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in range(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.uniform(low = 0.0, high = h)
            S[n + i * N, :] = T[i,:] + gap * dif[:]
    
    return S

def borderlineSMOTE(X, y, minority_target, N, k):
    """
    Returns synthetic minority samples.
    Parameters
    ----------
    X : array-like, shape = [n__samples, n_features]
        Holds the minority and majority samples
    y : array-like, shape = [n__samples]
        Holds the class targets for samples
    minority_target : value for minority class
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 
    h : high in random.uniform to scale dif of snythetic sample
    Returns
    -------
    safe : Safe minorities
    synthetic : Synthetic sample of minorities in danger zone
    danger : Minorities of danger zone

    onyl returns varitations of dangerous minoirty class options - hardest for clasifier to get
    """ 
    
    n_samples, _ = X.shape

    #Learn nearest neighbours on complete training set
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)
    
    safe_minority_indices = list()
    danger_minority_indices = list()
    
    for i in range(n_samples):
        if y[i] != minority_target: continue
        
        nn = neigh.kneighbors(X[i], return_distance=False)

        majority_neighbours = 0
        for n in nn[0]:
            if y[n] != minority_target:
                majority_neighbours += 1
                
        if majority_neighbours == len(nn):
            continue
        elif majority_neighbours < (len(nn)/2):
            safe_minority_indices.append(i)
        else:
            #DANGER zone
            danger_minority_indices.append(i)
            
    #SMOTE danger minority samples
    if len(danger_minority_indices) != 0:
        synthetic_samples = SMOTE(X[danger_minority_indices], N, k, h = 0.5)
        new_portion_labels = len(synthetic_samples)*[minority_target]
        l = np.insert(y,0,new_portion_labels)
        return np.concatenate((synthetic_samples,X)), l
    else:
        synthetic_samples = SMOTE(X[safe_minority_indices], N, k, h= 0.5)
        new_portion_labels = len(synthetic_samples)*[minority_target]
        l = np.insert(y,0,new_portion_labels)
        return np.concatenate((synthetic_samples,X)), l

if __name__ == "__main__":
    data = []
    l= []
    for i in range(0,2000):
        data.append(np.array([70+i]))
        l+=[1]
        if i%10 == 0:
            data.append(np.array([2+(i/1000)]))
            l+=[0]
    
    l = np.array(l)
    data = np.array(data)
    data = np.reshape(data,(-1,1))
    print(data)
    data,l = borderlineSMOTE(data,l,0,4000,4)
    print(data)