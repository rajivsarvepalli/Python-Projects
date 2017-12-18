import numpy as np
import warnings
class Ensemble:
    """
    Ensemble\n
    Creates a class that can contain multiple classifiers, and use them in conjunction to predict
    Parameters
    ----------
    *classifiers : as many classifiers as you want to be part of the Ensemble of learners, classifiers must contain a predict and fit method, the number of classifiers must be odd

    Attributes
    ----------
    classifiers : list of all the classifiers

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([0, 0, 0, 1, 1, 1])
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.naive_bayes import GaussianNB
    >>> ensemble = Ensemble(GaussianNB(),GaussianProcessClassifier(),RandomForestClassifier())
    >>> ensemble.fit(X, Y)
    >>> print(ensemble.predict([[-0.8, -1]]))
    [0]
    >>> print(ensemble.predict([[2.3, .9]]))
    [1]
    """
    def __init__(self,*classifiers):
        #make sure number of classifiers is odd
        if len(classifiers)%2 !=1:
            warnings.warn("Number of Classifiers should be odd")
        self.classifiers = classifiers
    def fit(self,X,y):
        """Fit Ensemble's classifiers with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        for x in self.classifiers:
            x.fit(X,y)
    def predict(self,X):
        """
        Perform classification on an array of test vectors X using combined classifiers
        AND classifier's output so an output of [1,1,0] for one data sample would result in a prediction of 1

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        pred =[]
        for x in self.classifiers:
            pred +=[x.predict(X)]
        pred =np.array(pred)
        #flip matrix around to make ANDing classifier's outputs easier
        pred = pred.transpose()
        predict = []
        for i in range(len(pred)):
            if sum(pred[i]>0)>len(self.classifiers)/2:
                predict+=[1]
            else: 
                predict +=[0]
        return predict
if __name__ == "__main__":
    #testing
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.utils import shuffle
    from sklearn.linear_model import LogisticRegression
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import time
    from gmuwork.shortcuts import ADASYN, quick_pfp1_file_reader, cluster_based_over_under_sampling, memory_usage_psutil, confusion_matrix_plotter, bar_stack_grapher
    ense = Ensemble(GaussianNB(),GaussianProcessClassifier(),RandomForestClassifier(),svm.SVC(),LogisticRegression())
    times = []
    start_time = time.time()
    v00 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:1000]
    v01 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter01")[0:1000]
    v10 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:1000]
    v11 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter01")[0:1000]
    v20 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:1000]
    v21 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter01")[0:1000]
    v30 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:1000]
    v31 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter01")[0:1000]
    v40 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:1000]
    v41 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter01")[0:1000]
    times+=[time.time()-start_time]
    d = np.concatenate((v00,v01,v10,v11,v20,v21,v30,v31,v40,v41))
    l =[0]*8000+[1]*2000
    data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.6,random_state=4) 
    start_time = time.time()
    data_train, labels_train = ADASYN(data_train,labels_train,ratio=2.0202,imb_threshold=.8,random_state=4)
    labels_train = np.array(labels_train)
    print(np.sum(labels_train>0))
    print(len(labels_train))
    labels_train, data_train = shuffle((data_train,labels_train),random_state=4)
    times+=[time.time()-start_time]
    start_time = time.time()
    ense.fit(data_train,labels_train)
    times+=[time.time()-start_time]
    print(memory_usage_psutil())
    start_time = time.time()
    pred = ense.predict(data_test)
    times+=[time.time()-start_time]
    bar_stack_grapher(times,['Times for Ensemble Testing(all parts) using clasifiers: (GaussianNB, GaussianProcess, RandomForestClassifier, Logisic Regression, SVC) '],['b','g','y','r'],legend_values=['Load Data','Oversampling Method(cluster)','Fitting Ensemble','Predicting with Ensemble'])    
    confusion_matrix_plotter(labels_test,pred,['Normal','Malicious'],title='Ensemble with (GaussianNB, GaussianProcess, RandomForestClassifier, Logisic Regression, SVC)')
    print(accuracy_score(labels_test,pred))
    print(classification_report(labels_test,pred,target_names=['Normal','Malicious']))
    print(times)


