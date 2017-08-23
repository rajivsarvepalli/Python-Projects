import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quick_pfp2_file_reader import alldata
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML, Image
import numpy as np
from sklearn.decomposition import PCA
fig, ax = plt.subplots()
s1 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State1")
s2 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State2")
s3 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State3")
s4 = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/State4")
sT = alldata("C:/Users/Rajiv Sarvepalli/Projects/Data for GMU/AllData/dataSet2/StateTamper")
c = np.concatenate((s1,s2,s3,s4,sT),axis=0)
np.random.shuffle(c)
s1,s2,s3,s4,sT = np.vsplit(c,5)
s1 =s1[0:10]
s2 =s2[0:10]
s3 =s3[0:10]
s4 =s4[0:10]
sT =sT[0:10]
d = np.concatenate((s1,s2,s3,s4,sT),axis=0)
print(memory_usage_psutil())
times = [x for x in range(0,len(s1[0]))]
ln, = plt.plot([], [], animated=True)
fig =plt.figure(1)
def animate(i):
    ln.set_data(times,d[i])
    print(i)
    if i >-1 and i<10:
        fig.suptitle("State1")
    elif i==10:
        fig.suptitle("State2")
        plt.draw()
        plt.pause(.00001)
    elif i==20:
        fig.suptitle("State3")
        plt.draw()
        plt.pause(.00001)
    elif i==30:
        fig.suptitle("State4")
        plt.draw()
        plt.pause(.00001)
    elif i==40:
        fig.suptitle("StateTamper")
        plt.draw()
        plt.pause(.00001)
    return ln,
anim = animation.FuncAnimation(fig, animate,frames=50, interval=2000, blit=True)
plt.xlim([0,len(times)])
plt.ylim([-1,1])
anim.save('C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/tests/animation(3).mp4', writer="ffmpeg")
print("Done")