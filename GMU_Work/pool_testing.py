from multiprocessing import Pool
import threading as thread
import multiprocessing as mp
import time
def f(a,b,c,d):
    return (a*b*c)**d
def poolrunner(a,b,c,d): 
    pool=mp.Pool()
    results = pool.starmap(f,zip(a,b,c,d))

a=[]
b=[]
c=[]
d=[]
for x in range(0,1000000):
    a+=[2]
    b+=[3]
    c+=[4]
    d+=[2]
start = time.time()
poolrunner(a,b,c,d)
print(time.time()-start)
start = time.time()
res =[]
for x in range(0,1000000):
    res.append(f(a[x],b[x],c[x],d[x]))
print(time.time()-start)
