#import hdf5storage
import numpy as np
import pickle
#import pprint
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import ndimage
import scipy
plt.rcParams["figure.figsize"] = (15,15)


# Defining the preprocessing functions
def low_pass_filter(data, N, f, fs= 2048):
    # f = f / (fs / 2)
    fnew=[x / (fs/2) for x in f]
    data = np.abs(data)
    b, a = signal.butter(N=N, Wn = fnew, btype="lowpass")
    output = signal.filtfilt(b, a, data,  axis=0)
    
    return output

def encode_mu_law(x,mu):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)
    return (fx+1)/2*mu+0.5

def windows(end, window_size, skip_step):
    """
    It creates a generator yielding windows with the length of the window_size.
    It starts to segment a sequence from the sample's start to its end.
 
    >>> gen = windows(1000, 200, 20)
    >>> next(gen)
    (0, 200)
    
    >>> next(gen)
    (20, 220)
    """
    start= 0
    while (start + window_size) <= end:
        yield start, start + window_size
        start += skip_step

def DataGenerator(sequence_matrix, window_size, skip_step):
    
    """Only sequences with the window_size are considered and no padding
    is used. This means we need to drop those length of which is below the window_size.
    
    >>> sequence_matrix = np.array([[c*i for c in range(1, 3)] for i in range(1, 5)])
    >>> gen = DataGenerator(sequence_matrix, 1, 2, 2)
    >>> next(gen)
    (array([[1, 2],
           [2, 4]]), 1)
    >>> next(gen)
    (array([[3, 6],
           [4, 8]]), 1)
    """
    num_features = sequence_matrix.shape[0]  # number of sequence elements
    for start, stop in windows(num_features, window_size, skip_step):
        yield (sequence_matrix[start:stop])

def loaddata_filt(mode,window_size,skip_step,idx):
    xf=[]
    xe=[]
    yf=[]
   
    num_gst=66
    num_rep=5
   
    for gst in range(0,num_gst):
        for rep in range(0,num_rep):
            if not(rep==1 and gst==33):
                
                with open('Final Dataset/Subj_{}/class{}{}/repetition{}/flexors.pkl'\
                          .format(idx,(gst+1)//10,(gst+1)%10,rep+1), 'rb') as f:
                    emg_sigf_final=pickle.load(f)
                
                with open('Final Dataset/Subj_{}/class{}{}/repetition{}/extensors.pkl'\
                          .format(idx,(gst+1)//10,(gst+1)%10,rep+1), 'rb') as f:
                    emg_sige_final=pickle.load(f)
            
                # emg_sigf_clipped=np.clip(emg_sigf_final,0,0.01)   
                emg_sigf_filtered=low_pass_filter(emg_sigf_final,N=1,f=[1])
                emg_sigf_clipped=np.clip(emg_sigf_filtered,0,0.1)
                # x_min = emg_sigf_clipped.min(axis=0, keepdims=True)
                # x_max = emg_sigf_clipped.max(axis=0, keepdims=True)
                # emg_sigf_norm1 = (emg_sigf_clipped - x_min)/(x_max-x_min)
                emg_sigf_norm = encode_mu_law(emg_sigf_clipped, mu=8)-4
          
            
                # emg_sige_clipped=np.clip(emg_sige_final,0,0.01)
                emg_sige_filtered=low_pass_filter(emg_sige_final,N=1,f=[1])
                emg_sige_clipped=np.clip(emg_sige_filtered,0,0.1)                                                          
                # x_min = emg_sige_clipped.min(axis=0, keepdims=True)
                # x_max = emg_sige_clipped.max(axis=0, keepdims=True)
                # emg_sige_norm1 = (emg_sige_clipped - x_min)/(x_max-x_min)
                emg_sige_norm = encode_mu_law(emg_sige_clipped, mu=8)-4
            
                # emg_sigt_norm=np.concatenate((emg_sigf_norm,emg_sige_norm),axis=2)[:,:,1::2]
                emg_sigt_norm=np.concatenate((emg_sigf_norm,emg_sige_norm),axis=2)[:,:,:]
                #channel 1 data
                #emg_sigt_norm1=emg_sigt_norm.reshape(-1,1,8*16)
                if rep+1 in [1,2,4,5] and mode=="train":
                
                    data_gen = DataGenerator(emg_sigt_norm, window_size, skip_step)
                    for index, sample in enumerate(data_gen):
                           
                            xf.append(sample)
                            yf.append(gst)
                         
                #data_gen = DataGenerator(emg_sige_norm, window_size, skip_step)
                #for index, sample in enumerate(data_gen):
                        #xe.append(sample)
                       
                elif rep+1 in [3] and mode=="test": 
                    
                    data_gen = DataGenerator(emg_sigt_norm, window_size, skip_step)
                    for index, sample in enumerate(data_gen):
                           
                            xf.append(sample)
                            yf.append(gst) 
                        
                #data_gen = DataGenerator(emg_sige_norm, window_size, skip_step)
                #for index, sample in enumerate(data_gen):
                        #xe.append(sample)
                         
                     
    return xf,yf,xe



