import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gmuwork.shortcuts import quick_pfp1_file_reader 
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML, Image
import numpy as np
from sklearn.decomposition import PCA
from gmuwork.shortcuts import moving_mean_smoothing
import time
from random import randint
def one_portion(data_row):
    from pandas import rolling_std,rolling_mean
    mov = rolling_std(data_row,200)
    means = []
    for i in range(0,len(mov),100):
        means.append(np.mean(mov[i:i+500]))
    index1 = np.argmax(means)
    return data_row[index1*100:(index1+1)*100]
def interesting_parts(data):
    new_data = []
    for x in data:
        new_data.append(one_portion(x))
    return np.array(new_data)
if __name__=="__main__":
   # fig, ax = plt.subplots()
    s1 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0000Path0000Iter00")[0:20]
    s2 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0001Path0001Iter00")[0:20]
    s3 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0002Path0002Iter00")[0:20]
    s4 = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0003Path0003Iter00")[0:20]
    sT = quick_pfp1_file_reader("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet3/Vector0004Path0004Iter00")[0:20]
    c = np.concatenate((s1,s2,s3,s4,sT),axis=0)
    c = moving_mean_smoothing(c,1000)
    start_time = time.time()
    print(time.time()-start_time)
    s1,s2,s3,s4,sT = np.vsplit(c,5)
    s1 =s1[0:10]
    s2 =s2[0:10]
    s3 =s3[0:10]
    s4 =s4[0:10]
    sT =sT[0:10]
    d = np.concatenate((s1,s2,s3,s4,sT),axis=0)
    times = [x for x in range(0,len(d[0]))]
    ln, = plt.plot([], [], animated=True)
    fig =plt.figure(1)
    def animate(i):
        ln.set_data(times,d[i])
        print(i)
        if i >-1 and i<10:
            fig.suptitle("State1_Vector000")
        elif i==10:
            fig.suptitle("State2_Vector110")
            plt.draw()
            plt.pause(.00001)
        elif i==20:
            fig.suptitle("State3_Vector220")
            plt.draw()
            plt.pause(.00001)
        elif i==30:
            fig.suptitle("State4_Vector330")
            plt.draw()
            plt.pause(.00001)
        elif i==40:
            fig.suptitle("StateTamper_Vector440")
            plt.draw()
            plt.pause(.00001)
        return ln,
    anim = animation.FuncAnimation(fig, animate,frames=50, interval=2000, blit=True)
    plt.xlim([0,len(times)])
    plt.ylim([-4,4])
    anim.save('C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/tests/smoothed_Data_animation.mp4', writer="ffmpeg")
    print("Done")