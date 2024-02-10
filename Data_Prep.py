import hdf5storage
import numpy as np
import pickle
#import pprint
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import ndimage
import scipy
plt.rcParams["figure.figsize"] = (15,15)



#  Loading the data from the original dataset
data=hdf5storage.loadmat('D:\Mansooreh\EMG_dataset\s17.mat')
emg_sigf=data['emg_flexors']
emg_sige=data['emg_extensors']
emg_class=data['class']
emg_adjclass=data['adjusted_class']
emg_adjrepet=data['adjusted_repetition']

# Detecting and deleting the rest states by differentiating the repetition signal
diff1=np.diff(emg_adjclass,axis=0)
ind1=(np.nonzero(diff1)[0])+1
emg_sigf_new=np.split(emg_sigf,ind1)[1::2]
emg_sige_new=np.split(emg_sige,ind1)[1::2]
emg_class_new=np.split(emg_adjclass,ind1)[1::2]

emg_sigf_new2=[]
emg_sige_new2=[]
for a in range(0,65+1):
    emg_sigf_new2.append(emg_sigf_new[a*5:(a+1)*5])
    emg_sige_new2.append(emg_sige_new[a*5:(a+1)*5])
    
# Now, the signal has a shape of (66,5,...,8,8) 

# Creating a distinct directory for the subject
for a in range(0,66):
    
    dirName1 = 'Final Dataset/Subj_17/class{}{}'.format((a+1)//10,(a+1)%10)
    os.makedirs(dirName1) 
    for b in range(0,5):
        
        dirName2 = 'Final Dataset/Subj_17/class{}{}/repetition{}'.format((a+1)//10,(a+1)%10,b+1)    
        os.makedirs(dirName2)   

# Saving the data in the directory        
for a in range(0,66):
    for b in range(0,5):
        with open('Final Dataset/Subj_17/class{}{}/repetition{}/flexors.pkl'\
                  .format((a+1)//10,(a+1)%10,b+1), 'wb') as f:
            pickle.dump(emg_sigf_new2[a][b], f)
                  
        with open('Final Dataset/Subj_17/class{}{}/repetition{}/extensors.pkl'\
                  .format((a+1)//10,(a+1)%10,b+1), 'wb') as f:
            pickle.dump(emg_sige_new2[a][b], f)
            

                                         
            