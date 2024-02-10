import torch
from Prep_Func import loaddata_filt
import numpy as np
import time
#import pickle
#import pprint
import os

def load_dataset(window_size,skip_step,idx):
    
    tr_sam=[]
    tr_class=[]
    ts_sam=[]
    ts_class=[]
    t_load1=time.time()
    tr_sam,tr_class,_=loaddata_filt('train',window_size,skip_step,idx)
    t_load2=time.time() 
    ts_sam,ts_class,_=loaddata_filt('test',window_size,skip_step,idx)
    t_load3=time.time() 
    
    return tr_sam,tr_class,ts_sam,ts_class,t_load1,t_load2,t_load3

def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders.
    dataloder_params = {'batch_size': 64,
                    'shuffle': True,
                    'num_workers': 3}
    '''
    
    tr_sam=[]
    tr_class=[]
    ts_sam=[]
    ts_class=[]                                          
    tr_sam1=[]
    tr_class1=[]
    ts_sam1=[]
    ts_class1=[]    
    tr_sam2=[]
    tr_class2=[]
    ts_sam2=[]
    ts_class2=[]
    
    tr_sam,tr_class,ts_sam,ts_class,t_load1,t_load2,t_load3=load_dataset(128,32,opt.sub_idx)
    
    with open(os.path.join(opt.exp, 'time_load_tr' + '.txt'), 'w') as f:
        f.write('%s\n' %(t_load2-t_load1))  
    with open(os.path.join(opt.exp,'time_load_ts' + '.txt'), 'w') as f:
        f.write('%s\n' %(t_load3-t_load2))  
        
    
    #num_workers=opt.num_workers)
    tr_sam1  = np.array(tr_sam)
    tr_class1 = np.array(tr_class)
    tr_sam2  = (torch.tensor(tr_sam1))
    tr_class2 = torch.IntTensor(tr_class1)
    
    train_dataset =torch.utils.data.TensorDataset(tr_sam2,tr_class2)
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=opt.batch_size,
                                                shuffle=opt.shuffle,
                                                num_workers=opt.num_workers)
    
    ts_sam1  = np.array(ts_sam)
    ts_class1 = np.array(ts_class)
    ts_sam2  = (torch.tensor(ts_sam1))
    ts_class2 = torch.IntTensor(ts_class1)
    
    test_dataset =torch.utils.data.TensorDataset(ts_sam2,ts_class2)
    ts_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=opt.batch_size_v,
                                                shuffle=opt.shuffle,
                                                num_workers=opt.num_workers)
    # return tr_dataloader
    return tr_dataloader,ts_dataloader

if __name__ == "__main__":
    import argparse
    '''
    Initialize opt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--shuffle', type=int, default=True) 
    parser.add_argument('--num_workers', type=int, default=12)
    options = parser.parse_args()

    options = parser.parse_args()
    tr_dataloader, val_dataloader, test_dataloader = init_dataset(options)
    print(tr_dataloader.__len__())
    print(val_dataloader.__len__())
    print(test_dataloader.__len__())






