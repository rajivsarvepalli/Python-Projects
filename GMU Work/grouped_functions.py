import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
class Daniels_algorithim:
    def __init__(self):
        '''
        instantiate 
        '''
        pass
    def compute_strangeness_distribution(self,X):
        '''
        baseline strangeness distribution
        '''
        n = NearestNeighbors(n_neighbors=5).fit(X)
        distances,indices = n.kneighbors(X)
        return np.sort(np.sum(distances,axis=1),axis=None)
    def compute_strangeness(self,concat_data,base1,base2,base3,base4):#assuming that strangeness is sum of distances not point-values
        '''
        strangeness of data with bases 

        '''
        b1 = NearestNeighbors(n_neighbors=5).fit(base1)
        b2 = NearestNeighbors(n_neighbors=5).fit(base2)
        b3 = NearestNeighbors(n_neighbors=5).fit(base3)
        b4 = NearestNeighbors(n_neighbors=5).fit(base4)
        strangeness =[]
        for x in concat_data:
            distance1,indices1= b1.kneighbors(x)
            distance2,indices2= b2.kneighbors(x)
            distance3,indices3= b3.kneighbors(x)
            distance4,indices4= b4.kneighbors(x)
            strangeness.append([np.sum(distance1),np.sum(distance2),np.sum(distance3),np.sum(distance4)])
        return strangeness
    def p_value_computation(self,concat_data,base1,base2,base3,base4):#p-values wrong
        strangeness_distribution = self.strangeness_distribution
        p_values =[]
        strangeness = np.array(self.compute_strangeness(concat_data,base1,base2,base3,base4))
        templ =[base1,base2,base3,base4]
        #pool = Pool(2)  #slower ?
        for x in strangeness:
                #res = pool.starmap(self.multihelper,zip(strangeness_distribution,templ,x))
                # res = pool.starmap(self.
                # multihelper,[(strangeness_distribution[0],base1,x[0]),(strangeness_distribution[1],base2,x[1]),(strangeness_distribution[2],base3,x[2]),(strangeness_distribution[3],base4,x[3])])
                # p_values.append(res)
                p1 = (sum(strangeness_distribution[0]>=x[0])+1)/(len(base1)+1)
                p2 = (sum(strangeness_distribution[1]>=x[1])+1)/(len(base2)+1)
                p3 = (sum(strangeness_distribution[2]>=x[2])+1)/(len(base3)+1)
                p4 = (sum(strangeness_distribution[3]>=x[3])+1)/(len(base4)+1)
                p_values.append([p1,p2,p3,p4])
        #pool.close()
        #pool.join()
        
        return np.array(p_values)
    def multihelper(self,strangeness_distributionv,base1,x):
        p1 = (sum(strangeness_distributionv>=x)+1)/(len(base1)+1)
        return p1
    def comparing_with_c(self,ci,concat_data):
        base1 = self.base1
        base2 = self.base2
        base3 = self.base3
        base4 = self.base4
        c=ci
        p_values= self.p_value_computation(concat_data,base1,base2,base3,base4)
        predict =[]
        for x in p_values:
            if sum(x<=(1-c))==4:
                predict+=[1]
            else:
                predict+=[0]
        return np.array(predict)
    def fit(self,X):
        index = len(X)//4
        base1 = X[0:index]
        base2 = X[index:index*2]
        base3 = X[index*2:index*3]
        base4 = X[index*3:len(X)]
        self.strangeness_distribution= np.array([self.compute_strangeness_distribution(base1),self.compute_strangeness_distribution(base2),self.compute_strangeness_distribution(base3),self.compute_strangeness_distribution(base4)])
        self.base1 = base1
        self.base2 = base2
        self.base3 = base3
        self.base4 = base4
    def predict(self,X,c):
        return self.comparing_with_c(c,X)


def load_pfp(directory,label,pfp2 =True):
    '''
    loads pfp files into memory \n
    defualtly chose file_type as pfp2, set pfp2 = False for pfp1 files\n
    returns data, labels
    '''
    import os
    def extract_pfp1(file_name):
        f= open(file_name,"rb")
        f.seek(50)
        a = np.fromfile(f,dtype=np.single)
        return a
    def extract_pfp2(file_name):
        f= open(file_name,"r")
        a = np.fromfile(f,dtype=np.single)
        data =a[14:len(a)]
        return data
    labels =[]
    if pfp2:
        extract_func = extract_pfp2
    else:
        extract_func = extract_pfp1
    files=[]
    for infile in os.listdir(directory):
        labels+=[label]
        files.append(extract_func(directory + "/" + infile))
    return np.array(files),labels
def train_test_split(data,labels,train_size=.5):
    '''
    Splits data set according to the size set while maintaining corresponding labels\n
    Returns x_train, y_train, x_test, y_test
    '''
    labels = np.array(labels).reshape(len(data),1)
    comb = np.append(data,labels,axis=1)
    np.random.shuffle(comb)
    size = int(len(comb)*train_size)
    return comb[0:size,0:len(comb[0])-1],comb[0:size,-1],comb[size:len(comb),0:len(comb[0])-1],comb[size:len(comb),-1]
def PCA(*args,n_comp=None):
    '''
    takes a tuple of arrays and applies pca to alll of them
    returns concatenated version of array with pca applied
    '''
    from sklearn.decomposition import PCA as pca
    d = np.concatenate(*args,axis=0)
    pc =pca(n_components=n_comp)
    return pc.fit_transform(d)
# def strangeness(setA,setB):#assuming that Set A is baselines
#     '''
#     assuming SetA is baseline, and setB is concatanted data\n
#     returns strangeness of baseline, strangeness of test data
#     '''
#     from sklearn.neighbors import NearestNeighbors
#     import warnings
#     warnings.filterwarnings("ignore")
#     np.random.shuffle(setA)
#     i = len(setA)
#     index = len(setA)//4
#     s1 = setA[0:index]
#     s2 = setA[index:index*2]
#     s3 = setA[index*2:index*3]
#     s4 = setA[index*3:len(setA)]
#     da = Daniels_algorithim()
#     da.fit()
#     return da.compute_strangeness_distribution(setA),da.compute_strangeness(setB,s1,s2,s3,s4)
    

    
# def normal_or_not(strangeA,strangeM):
#     '''
#     arguments are: baseline strangeness followed by test_data strangeness
#     '''
#     import warnings
#     warnings.filterwarnings("ignore")
#     np.random.shuffle(strangeA)
#     index = len(strangeA)//4
#     s1 = strangeA[0:index]
#     s2 = strangeA[index:index*2]
#     s3 = strangeA[index*2:index*3]
#     s4 = strangeA[index*3:len(strangeA)]
#     p_values =[]
#     c=.98
#     for x in strangeM:
#         p1 = (sum(s1>=x[0])+1)/(len(s1)+1)
#         p2 = (sum(s2>=x[1])+1)/(len(s2)+1)
#         p3 = (sum(s3>=x[2])+1)/(len(s3)+1)
#         p4 = (sum(s4>=x[3])+1)/(len(s4)+1)
#         p_values.append([p1,p2,p3,p4])
#     p_values = np.array(p_values)
#     predict =[]
#     for x in p_values:
#         val = sum((1-c)>=x)
#         if val==4:
#             predict+=[1]
#         else:
#             predict+=[0]
#     return np.array(predict)
def tpr_fpr_calc(y_true,y_predict):
    '''
    Returns fpr, tpr
    '''
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true,y_predict)
    fpr = (1-(cm[0][0]/(cm[0][0]+cm[0][1])))
    tpr =(cm[1][1]/(cm[1][0]+cm[1][1]))
    return fpr,tpr
def roc_curve(fpr,tpr,file_path):
    '''
    saves the ROC curve to a the specific file given multiple tpr and fpr values
    '''
    import matplotlib.pyplot as plt
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-.02,1.02])
    plt.ylim([0,1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig(file_path)
if __name__ =="__main__":
    from sklearn.metrics import accuracy_score
    s1,l1 = load_pfp("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State1",0)
    s2,l2 = load_pfp("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State2",0)
    s3,l3 = load_pfp("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State3",0)
    s4,l4 = load_pfp("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State4",0)
    sT,lT = load_pfp("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/StateTamper",1)
    l =np.concatenate((l1,l2,l3,l4))
    pca = PCA((s1,s2,s3,s4,sT),n_comp=195)
    sT = pca[780:len(pca)]
    pca = pca[0:780]
    x_train,y_train,x_test,y_test = train_test_split(pca,l,train_size=.8)
    da = Daniels_algorithim()
    da.fit(pca)
    pred = da.predict(np.concatenate((x_test,sT)),.99)
    print(accuracy_score(np.concatenate((y_test,lT),axis=0),pred))