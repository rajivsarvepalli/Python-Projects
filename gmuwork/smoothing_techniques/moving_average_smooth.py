import numpy as np
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    new_x = (cumsum[N:] - cumsum[:-N]) / float(N)
    last_values =[]
    for i in range(1,N):
        last_portion = x[len(x)-2*N+i:len(x)]
        one_value = (np.cumsum(last_portion)/float(N-1))[N-1]
        last_values +=[one_value]

    return np.append(new_x,last_values)
def smooth2D(X,m):
    smoothed_curves =[]
    for x in X:
        smoothed_curves.append(running_mean(x,m))
    return np.array(smoothed_curves)
if __name__ == "__main__":
    from random import randint
    import matplotlib.pyplot as plt
    data =[]
    for i in range(0,100):
        data+=[i]
    for i in range(0,5):
        index = randint(0,len(data)-1)
        data[index] = data[index]- randint(0,100)
    x =[]
    for i in range(0,100):
        x+=[i]
    plt.figure()
    m=10
    data = np.array([data,data])
    new_data = smooth2D(data,m)
    plt.figure()
    plt.plot(x,new_data)
    plt.show()

    