import pyfftw
import numpy as np
from scipy.fftpack import fft,ifft
import time
def pyfft(a):
    a = pyfftw.n_byte_align(a,None)
    start_time = time.time()
    pyfftw.interfaces.scipy_fftpack.fft(a,threads=2)
    print(time.time()-start_time)
def pyifft(a):
    a =pyfftw.interfaces.scipy_fftpack.fft(a)
    start_time = time.time()
    a = pyfftw.n_byte_align(a,None)
    pyfftw.interfaces.scipy_fftpack.ifft(a)
    print(time.time()-start_time)
def pyfft_align(a):
    start_time = time.time()
    a = pyfftw.n_byte_align(a,None)
    pyfftw.interfaces.scipy_fftpack.fft(a)
    print(time.time()-start_time)
def scipyfft(a):
    start_time = time.time()
    fft(a)
    print(time.time()-start_time)
def scipyifft(a):
    a = fft(a)
    start_time = time.time()
    ifft(a)
    print(time.time()-start_time)
if __name__ =="__main__":
    a = np.array([5,182,2]*256)
    pyfftw.interfaces.cache.enable()
    # pyfft(a)
    # scipyfft(a)
    print('start')
    st = time.time()
    for i in range(0,300000):
        np.sum(a)
    print(time.time()-st)    
