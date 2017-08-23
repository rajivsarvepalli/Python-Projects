import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import time
from sklearn.cross_validation import train_test_split
from quick_pfp2_file_reader import alldata
from quick_pfp1_file_reader import alldata as alld
from quick_txt_reader import allData
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
from multiprocessing import Pool
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
            #strangeness.append([np.sum(base1[indices1]),np.sum(base2[indices2]),np.sum(base3[indices3]),np.sum(base4[indices4])])
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
    def fit(self,base1,base2,base3,base4):
        self.strangeness_distribution= np.array([self.compute_strangeness_distribution(base1),self.compute_strangeness_distribution(base2),self.compute_strangeness_distribution(base3),self.compute_strangeness_distribution(base4)])
        self.base1 = base1
        self.base2 = base2
        self.base3 = base3
        self.base4 = base4
    def predict(self,X,c):
        return self.comparing_with_c(c,X)


    #testing
def test1():#this dataset prefers 98
    '''runs with dataset1 (the one with txt files)
    this dataset prefers conficnec level of 98
    '''
    times=[]
    start_time = time.time()
    a = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeA")
    b = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeB")
    c = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeC")
    d = allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeD")
    m =allData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/KMeans_training/ModeM")
    times+=[time.time()-start_time]
    #optional: randomzing bases
    e = np.concatenate((a,b,c,d),axis=0)
    np.random.shuffle(e)
    index = len(e)//4
    a = e[0:index]
    b = e[index:index*2]
    c = e[index*2:index*3]
    d = e[index*3:len(e)]
    #end randomizing(optional)
    pca = PCA(n_components=195)
    start_time =time.time()
    f = np.concatenate((a,b,c,d,m),axis=0)
    f = pca.fit_transform(f)
    a,b,c,d,m = np.split(f,5)
    times+=[time.time()-start_time]
    l =[0]*312+[1]*78
    start_time = time.time()
    da = Daniels_algorithim()
    da.fit(a[0:30],a[0:30],a[0:30],a[0:30])
    data_train, data_test, labels_train, labels_test=train_test_split(f,l,train_size =0.8,random_state=4)
    x = da.predict(data_test,.98)
    times+=[time.time()-start_time]
    start_time = time.time()
    roc_curve(da,data_test,labels_test)
    times+=[time.time()-start_time]
    timegrapher(times)
    print(accuracy_score(labels_test,x))
def test2():#dataset with fewer files many features
    times=[]
    start_time = time.time()
    s1 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State1")
    s2 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State2")
    s3 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State3")
    s4 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/State4")
    sT = alldata("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet2/StateTamper")
    times+=[time.time()-start_time]
    l =[0]*780 #(780)
    c = np.concatenate((s1,s2,s3,s4),axis=0)
    np.random.shuffle(c)
    s1 = c[0:195]
    s2 = c[195:390]
    s3 = c[390:585]
    s4 = c[585:780]
    pca = PCA(n_components=5000)
    start_time =time.time()
    d = np.concatenate((s1,s2,s3,s4,sT),axis=0)
    d = pca.fit_transform(d)
    sT = d[780:len(d)]
    d = d[0:780]
    np.random.shuffle(d)
    s1 = d[0:160]
    s2 = d[160:320]
    s3 = d[320:480]
    s4 = d[480:640]
    times+=[time.time()-start_time]
    #optional: randomzing bases
    
    #end randomizing(optional)
    start_time =time.time()
    da = Daniels_algorithim()
    data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.8,random_state=4)
    da.fit(s1,s2,s3,s4)
    x = da.predict(np.concatenate((data_test,sT)),.99)
    print(accuracy_score(np.concatenate((labels_test,[1]*195)),x))
    times+=[time.time()-start_time]
    start_time = time.time()
    roc_curve(da,data_test,labels_test)
    times+=[time.time()-start_time]
    from shortcuts import bar_stack_grapher
    bar_stack_grapher(times,["TEst1"],['b','g','y','r'],legend_values=['REad','PCA','1 Run','ROC'])
    print(x)
def test3():#dataset with many files few features pfp1
    times =[]
    st = time.time()
    v00 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0000Path0000Iter00")
    v01 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0000Path0000Iter01")
    v10 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0001Path0001Iter00")
    v11 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0001Path0001Iter01")
    v20 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0002Path0002Iter00")
    v21 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0002Path0002Iter01")
    v30 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0003Path0003Iter00")
    v31 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0003Path0003Iter01")
    v40 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0004Path0004Iter00")
    v41 = alld("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/dataSet3/Vector0004Path0004Iter01")
    times+=[time.time()-st]
    st =time.time()
    pca = PCA(n_components=200)
    v00 = pca.fit_transform(v00)
    v01 = pca.fit_transform(v01)
    v10 = pca.fit_transform(v10)
    v11 = pca.fit_transform(v11)
    v20 = pca.fit_transform(v20)
    v21 = pca.fit_transform(v21)
    v30 = pca.fit_transform(v30)
    v31 = pca.fit_transform(v31)
    v40 = pca.fit_transform(v40)
    v41 = pca.fit_transform(v41)
    times+=[time.time()-st]
    d = np.concatenate((v00,v01,v10,v11,v20,v21,v30,v31,v40,v41),axis=0)
    l =[0]*39949+[1]*9988
    data_train, data_test, labels_train, labels_test=train_test_split(d,l,train_size =0.8,random_state=4)
    st = time.time()
    da = Daniels_algorithim()
    da.fit(np.concatenate((v00,v01),axis=0),np.concatenate((v10,v11),axis=0),np.concatenate((v20,v21),axis=0),np.concatenate((v30,v31),axis=0))
    x = da.predict(data_test,.99)
    times+=[time.time()-st]
    timegrapher(times)
    print(accuracy_score(labels_test,x))
    print(x)
def roc_curve(da,data_test,labels_test):
    fpr =[]
    tpr =[]
    for i in range(0,100):
        x = da.predict(data_test,(i/100))
        cm = confusion_matrix(labels_test,x)
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
    for i in range(0,95,1):
         plt.annotate(str(i),(fpr[i],tpr[i]))
    for i in range(97,100,1):
         plt.annotate(str(i),(fpr[i],tpr[i]))
         #print(str(i)+ " : " + str(fpr[i])+ ","+str(tpr[i]))
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-.02,1.02])
    plt.ylim([0,1.4])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver operating characteristic Dataset2') #name of current dataset that you are testing
    #plt.show()
def timegrapher(times):
    plt.figure()
    objects = ('Read', 'PCA', '1 Run','ROC')  #remove ROC if doing test3
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, times, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Time (in seconds)')
    plt.title('Time of major functions (Dataset3)')#name of current dataset that you are testing
    for i, v in enumerate(times):
        plt.text(i-.4, v, "%.5f" % (v), color='black', fontweight='bold')
    plt.show()

#call tests here
if __name__=="__main__":
    test2()