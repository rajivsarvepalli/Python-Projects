import numpy as np
def get_bases(array):
    base1 =[]
    base2 =[]
    base3 =[]
    base4 = []
    baseT = []
    testl = np.concatenate((r[0],r[1],r[2],r[3]))
    base1,base2,base3,base4,baseT = np.split(testl,5)
    base1,base2,base3,base4,baseT = remove_every_other([base1,base2,base3,base4,baseT])
    return base1,base2,base3,base4,baseT
def remove_every_other(array):
    for i in range(0,len(array)):
        array[i] = array[i][0::2]
        array[i] = np.reshape(array[i],(78,19744))
    return np.array(array)
if __name__ == "__main__":
    r = np.load(r"C:\Users\Rajiv Sarvepalli\Projects\Data for GMU\tests\Matrix_Profile_of_dataset1.npy")
    base1,base2,base3,base4,baseT = get_bases(r)
    print('base1',np.sum(base1))
    print('base2',np.sum(base2))
    print('base3',np.sum(base3))
    print('base4',np.sum(base4))
    print('baseT',np.sum(baseT))
    base1_sums = np.sum(base1,axis=1)
    base2_sums = np.sum(base2,axis=1)
    base3_sums = np.sum(base3,axis=1)
    base4_sums = np.sum(base4,axis=1)
    baseT_sums = np.sum(baseT,axis=1)
    for i in range(0,len(base1_sums)):
        base4_sums[i] = max(base1_sums[i],base2_sums[i],base3_sums[i],base4_sums[i],baseT_sums[i])
    truth = baseT_sums>base4_sums
    print(truth)
